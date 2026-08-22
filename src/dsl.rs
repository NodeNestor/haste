/// The line protocol. One verb per line; multi-line payloads (E/I/N) end with a
/// lone "." line ("..." style escaping: a payload line that must literally start
/// with '.' is written with one extra leading dot).
///
///   R <id|path> [a:b]      read (numbered lines)
///   E <id> <a>:<b>         replace lines a..=b, payload follows
///   I <id> <a>             insert after line a (0 = top), payload follows
///   N <path>               create file, payload follows
///   G <regex> [id|path]    search
///   X <shell...>           run command
///   A <profile> <task...>  subagent
///   D <message...>         done (rest of stream joins the message)
///   <cfg verb> <args...>   config-declared tool
#[derive(Debug, Clone, PartialEq)]
pub enum Cmd {
    Read { target: String, range: Option<(usize, usize)> },
    Edit { target: String, a: usize, b: usize, body: String },
    Insert { target: String, after: usize, body: String },
    New { path: String, body: String },
    Grep { pat: String, target: Option<String> },
    Exec { line: String },
    Agent { profile: String, task: String },
    Done { msg: String },
    /// View an image file: attached to the next model request.
    View { target: String },
    /// Say something to the user without ending the run.
    Say { text: String },
    /// Signature outline of a file or directory (codemap).
    Outline { target: String },
    /// Search/replace edit (cc-json dialect: the Edit tool's old/new strings).
    Replace { target: String, old: String, new: String },
    Custom { verb: char, args: String },
}

enum Pending {
    Edit { target: String, a: usize, b: usize },
    Insert { target: String, after: usize },
    New { path: String },
}

/// Feed streamed model output in arbitrary chunks; complete commands come out
/// the moment their final byte arrives, so tools can run while the model is
/// still generating the rest of the message.
pub struct Lexer {
    buf: String,
    pending: Option<(Pending, Vec<String>)>,
    done_msg: Option<String>,
    /// Accept Claude-Code-style {"tool":...,"input":...} lines as commands.
    cc_json: bool,
}

impl Default for Lexer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lexer {
    pub fn new() -> Lexer {
        Lexer { buf: String::new(), pending: None, done_msg: None, cc_json: false }
    }

    pub fn new_dialect(cc_json: bool) -> Lexer {
        Lexer { cc_json, ..Lexer::new() }
    }

    pub fn feed(&mut self, chunk: &str, out: &mut Vec<Cmd>) {
        self.buf.push_str(chunk);
        while let Some(nl) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=nl).collect();
            self.line(line.trim_end_matches(['\n', '\r']), out);
        }
    }

    /// Flush at end of stream: a trailing line without newline still counts,
    /// and an unterminated payload is applied as-is (model ran out of tokens).
    pub fn finish(&mut self, out: &mut Vec<Cmd>) {
        if !self.buf.is_empty() {
            let line = std::mem::take(&mut self.buf);
            self.line(line.trim_end_matches('\r'), out);
        }
        if let Some((p, body)) = self.pending.take() {
            out.push(close_payload(p, body));
        }
        if let Some(msg) = self.done_msg.take() {
            out.push(Cmd::Done { msg: msg.trim().to_string() });
        }
    }

    fn line(&mut self, line: &str, out: &mut Vec<Cmd>) {
        if let Some(msg) = &mut self.done_msg {
            msg.push('\n');
            msg.push_str(line);
            return;
        }
        if let Some((_, body)) = &mut self.pending {
            if line == "." {
                let (p, body) = self.pending.take().unwrap();
                out.push(close_payload(p, body));
            } else if let Some(rest) = line.strip_prefix('.') {
                body.push(rest.to_string()); // "..x" -> ".x", ". x" stays ". x"? no: strip one dot
            } else {
                body.push(line.to_string());
            }
            return;
        }
        let line = line.trim_start();
        if line.is_empty() {
            return;
        }
        if self.cc_json {
            // Accept every notation the model produces, including transcript
            // echoes with diffusion block-glitched prefixes ("ISTANT (tool
            // call) Edit input={..."): find " input={", the word before it is
            // the tool name, everything before that is noise.
            if let Some(idx) = line.find(" input={") {
                let name = line[..idx].rsplit([' ', ')']).next().unwrap_or("");
                if !name.is_empty() {
                    if let Some(cmd) = parse_cc_tool_named(name, &line[idx + 7..]) {
                        out.push(cmd);
                    }
                    return;
                }
            }
            if line.starts_with('{') {
                if let Some(cmd) = parse_cc_tool(line) {
                    out.push(cmd);
                }
                return;
            }
        }
        let mut it = line.splitn(2, ' ');
        let verb = it.next().unwrap_or("");
        let rest = it.next().unwrap_or("").trim();
        if verb.len() != 1 {
            return; // prose/noise line: ignored
        }
        let v = verb.chars().next().unwrap();
        match v {
            'R' => {
                let mut p = rest.splitn(2, ' ');
                let target = p.next().unwrap_or("").to_string();
                let range = p.next().and_then(parse_range);
                if !target.is_empty() {
                    out.push(Cmd::Read { target, range });
                }
            }
            'E' => {
                let mut p = rest.splitn(2, ' ');
                let target = p.next().unwrap_or("").to_string();
                if let Some((a, b)) = p.next().and_then(parse_range) {
                    self.pending = Some((Pending::Edit { target, a, b }, Vec::new()));
                }
            }
            'I' => {
                let mut p = rest.splitn(2, ' ');
                let target = p.next().unwrap_or("").to_string();
                if let Some(after) = p.next().and_then(|s| s.trim().parse().ok()) {
                    self.pending = Some((Pending::Insert { target, after }, Vec::new()));
                }
            }
            'N' => {
                if !rest.is_empty() {
                    self.pending = Some((Pending::New { path: rest.to_string() }, Vec::new()));
                }
            }
            'G' => {
                let (pat, target) = split_pattern(rest);
                if !pat.is_empty() {
                    out.push(Cmd::Grep { pat, target });
                }
            }
            'X' => {
                if !rest.is_empty() {
                    out.push(Cmd::Exec { line: rest.to_string() });
                }
            }
            'A' => {
                let mut p = rest.splitn(2, ' ');
                let profile = p.next().unwrap_or("").to_string();
                let task = p.next().unwrap_or("").trim().to_string();
                if !profile.is_empty() && !task.is_empty() {
                    out.push(Cmd::Agent { profile, task });
                }
            }
            'V' => {
                if !rest.is_empty() {
                    out.push(Cmd::View { target: rest.to_string() });
                }
            }
            'S' => {
                if !rest.is_empty() {
                    out.push(Cmd::Say { text: rest.to_string() });
                }
            }
            'O' => {
                out.push(Cmd::Outline { target: rest.to_string() });
            }
            'D' => {
                self.done_msg = Some(rest.to_string());
            }
            c if c.is_ascii_uppercase() => {
                out.push(Cmd::Custom { verb: c, args: rest.to_string() });
            }
            _ => {}
        }
    }
}

/// Map a Claude-Code tool call line onto haste commands. Unknown tools are
/// ignored (the empty-turn nudge handles a message of only unknowns).
fn parse_cc_tool(line: &str) -> Option<Cmd> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let tool = v["tool"].as_str().or_else(|| v["name"].as_str())?.to_string();
    let inp = if v["input"].is_object() { v["input"].clone() } else { v["arguments"].clone() };
    map_cc_tool(&tool, &inp)
}

/// "Name input={json}" transcript form: the json IS the input object.
fn parse_cc_tool_named(name: &str, json_part: &str) -> Option<Cmd> {
    let inp: serde_json::Value = serde_json::from_str(json_part).ok()?;
    map_cc_tool(name, &inp)
}

fn map_cc_tool(tool: &str, inp: &serde_json::Value) -> Option<Cmd> {
    let s = |k: &str| inp[k].as_str().map(str::to_string);
    match tool {
        "Bash" => Some(Cmd::Exec { line: s("command")? }),
        "Read" => {
            let target = s("file_path").or_else(|| s("path"))?;
            let range = match (inp["offset"].as_u64(), inp["limit"].as_u64()) {
                (Some(o), Some(l)) => Some((o as usize, (o + l - 1) as usize)),
                (Some(o), None) => Some((o as usize, usize::MAX / 2)),
                _ => None,
            };
            Some(Cmd::Read { target, range })
        }
        "Write" => Some(Cmd::New { path: s("file_path").or_else(|| s("path"))?, body: s("content")? }),
        "Edit" => Some(Cmd::Replace {
            target: s("file_path").or_else(|| s("path"))?,
            old: s("old_string")?,
            new: s("new_string")?,
        }),
        "Grep" => Some(Cmd::Grep { pat: s("pattern")?, target: s("path") }),
        "Glob" => Some(Cmd::Grep { pat: s("pattern")?, target: s("path") }),
        "Task" => Some(Cmd::Agent { profile: "researcher".into(), task: s("prompt")? }),
        _ => None,
    }
}

fn close_payload(p: Pending, body: Vec<String>) -> Cmd {
    let body = body.join("\n");
    match p {
        Pending::Edit { target, a, b } => Cmd::Edit { target, a, b, body },
        Pending::Insert { target, after } => Cmd::Insert { target, after, body },
        Pending::New { path } => Cmd::New { path, body },
    }
}

fn parse_range(s: &str) -> Option<(usize, usize)> {
    let (a, b) = s.trim().split_once(':')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// `G pattern [target]` — pattern may be "quoted with spaces" or a bare word.
fn split_pattern(rest: &str) -> (String, Option<String>) {
    if let Some(r) = rest.strip_prefix('"') {
        if let Some(end) = r.find('"') {
            let pat = r[..end].to_string();
            let tgt = r[end + 1..].trim();
            return (pat, (!tgt.is_empty()).then(|| tgt.to_string()));
        }
    }
    let mut p = rest.splitn(2, ' ');
    let pat = p.next().unwrap_or("").to_string();
    let tgt = p.next().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    (pat, tgt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(input: &str) -> Vec<Cmd> {
        let mut lx = Lexer::new();
        let mut out = Vec::new();
        // feed in awkward 3-byte chunks to prove streaming works
        let bytes = input.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let j = (i + 3).min(bytes.len());
            lx.feed(std::str::from_utf8(&bytes[i..j]).unwrap_or(""), &mut out);
            i = j;
        }
        lx.finish(&mut out);
        out
    }

    #[test]
    fn parses_batch() {
        let cmds = all("R 3 10:20\nG \"fn main\" src/\nX cargo check\n");
        assert_eq!(cmds.len(), 3);
        assert_eq!(cmds[0], Cmd::Read { target: "3".into(), range: Some((10, 20)) });
        assert_eq!(cmds[1], Cmd::Grep { pat: "fn main".into(), target: Some("src/".into()) });
    }

    #[test]
    fn payload_and_escape() {
        let cmds = all("E 2 5:6\nlet x = 1;\n..hidden\n.\nD fixed\n");
        assert_eq!(
            cmds[0],
            Cmd::Edit { target: "2".into(), a: 5, b: 6, body: "let x = 1;\n.hidden".into() }
        );
        assert_eq!(cmds[1], Cmd::Done { msg: "fixed".into() });
    }

    #[test]
    fn cc_dialect_accepts_all_three_notations() {
        let mut lx = Lexer::new_dialect(true);
        let mut out = Vec::new();
        lx.feed(
            "{\"tool\":\"Bash\",\"input\":{\"command\":\"ls\"}}\n\
             ASSISTANT (tool call) Read input={\"file_path\":\"a.py\"}\n\
             Grep input={\"pattern\":\"foo\"}\n",
            &mut out,
        );
        lx.finish(&mut out);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[0], Cmd::Exec { line: "ls".into() });
        assert_eq!(out[1], Cmd::Read { target: "a.py".into(), range: None });
        assert_eq!(out[2], Cmd::Grep { pat: "foo".into(), target: None });
    }

    #[test]
    fn ignores_prose() {
        let cmds = all("Sure, let me look at that.\nR 1\n");
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn unterminated_payload_applies_on_finish() {
        let cmds = all("N src/new.rs\nfn f() {}");
        assert_eq!(cmds[0], Cmd::New { path: "src/new.rs".into(), body: "fn f() {}".into() });
    }
}
