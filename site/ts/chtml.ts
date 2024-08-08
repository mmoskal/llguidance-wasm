export type AnyInput =
  | HTMLInputElement
  | HTMLTextAreaElement
  | HTMLSelectElement;

let errElt: HTMLDivElement;
let progressElt: HTMLDivElement;
let _rootElt: HTMLElement;

export function rootElt() {
  if (!_rootElt) {
    _rootElt = document.getElementById("app") || document.body;
  }
  return _rootElt;
}

export function elt(id: string) {
  const r = document.getElementById(id);
  if (!r) throw new Error(`element ${id} not found`);
  return r;
}

export function setError(msg: string) {
  if (!errElt) {
    errElt = div("error");
    rootElt().prepend(errElt);
  }
  errElt.textContent = msg;
}

export function setProgress(msg: string) {
  if (!progressElt) {
    progressElt = div("progress");
    rootElt().prepend(progressElt);
  }
  progressElt.textContent = msg;
}

export function setupFileDrop(
  handler: (b64: string, name: string) => void,
  exts: string[]
) {
  function handleFile(f: File) {
    const name = f.name.toLowerCase();
    if (!exts.some((e) => name.endsWith(e))) {
      setError(`Unexpected file type (${name}); expecting ${exts.join(", ")}`);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const str = e.target?.result;
      if (typeof str == "string") {
        const idx = str.indexOf(";base64,");
        if (idx > 0) {
          const b64 = str.slice(idx + 8);
          handler(b64, name);
          return;
        }
      }
      setError("can't load file");
    };
    reader.readAsDataURL(f);
  }

  document.body.ondragover = dragOverHandler;
  document.body.ondrop = dropHandler;

  function dragOverHandler(ev: Event) {
    ev.preventDefault();
  }

  function dropHandler(ev: DragEvent) {
    if (!ev.dataTransfer) return;
    ev.preventDefault();
    const files = ev.dataTransfer.files;
    if (files.length != 1) setError("one file expected");
    else for (let i = 0; i < files.length; ++i) handleFile(files[i]);
  }
}

export function btn(lbl: string, cls: string, onclick: () => void) {
  const b = document.createElement("button");
  b.textContent = lbl;
  b.className = "pure-button " + cls;
  b.type = "button";
  b.onclick = (ev) => {
    ev.preventDefault();
    onclick();
  };
  return b;
}

export function btnLink(lbl: string, url: string, { newTab = false } = {}) {
  const b = document.createElement("a");
  b.textContent = lbl;
  b.className = "pure-button";
  b.href = url;
  if (newTab) b.target = "_blank";
  return b;
}

export function setPrim(b: HTMLButtonElement, prim: boolean) {
  if (prim) b.classList.add("pure-button-primary");
  else b.classList.remove("pure-button-primary");
}

export function div(cls = "") {
  const d = document.createElement("div");
  d.className = cls;
  return d;
}

export function text(t: string, elt = "span") {
  const [tag, cls] = elt.split(".", 2);
  const s = document.createElement(tag);
  if (cls) s.className = cls;
  s.textContent = t;
  return s;
}

export function append(par: HTMLElement, ...elts: (string | HTMLElement)[]) {
  for (const elt of elts) {
    if (elt == null) continue;
    if (typeof elt == "string") par.appendChild(text(elt));
    else par.appendChild(elt);
  }
}

async function httpCheckError(resp: Response) {
  if (resp.status != 200) {
    const msg = `error ${resp.url} -> ${resp.status}`;
    setError(msg); // set error first, in case getting text() fails
    let txt = await resp.text();
    if (/^</.test(txt))
      txt = txt.replace(/<[^>]+>/g, "").replace(/\&copy[^]*/, "");
    setError(msg + ": " + txt);
    throw new Error(msg);
  }
}

export async function getJSON(path: string) {
  const resp = await fetch(path);
  await httpCheckError(resp);
  return await resp.json();
}

export async function postJSON(path: string, data: any, method = "post") {
  setError("");

  let resp: Response;
  if (data instanceof Uint8Array) {
    resp = await fetch(path, {
      method,
      headers: {
        "content-type": "application/octet-stream",
        "content-length": "" + data.byteLength,
      },
      body: data,
    });
  } else {
    resp = await fetch(path, {
      method,
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(data),
    });
  }

  await httpCheckError(resp);
  return await resp.json();
}

export function mkElt(tag: string, ...children: (string | HTMLElement)[]) {
  const m = /(\S+)\s+(.*)/.exec(tag);
  let cls = "";
  if (m) {
    tag = m[1];
    cls = m[2];
  }
  const r = document.createElement(tag);
  if (cls) r.className = cls;
  append(r, ...children);
  return r;
}

export function mkLink(href: string, text: string, cls = "pure-menu-link") {
  const anch = document.createElement("a");
  anch.href = href;
  anch.textContent = text;
  anch.className = cls;
  return anch;
}

export function mkTable(body: HTMLElement, headers: string[] = ["#", "Name"]) {
  const tbody = mkElt("tbody");
  let tr: HTMLElement;
  body.appendChild(
    mkElt(
      "table pure-table pure-table-horizontal",
      mkElt("thead", (tr = mkElt("tr", ...headers.map((h) => mkElt("th", h))))),
      tbody
    )
  );

  return { tr, tbody };
}

export function timeStr(n: number) {
  if (!n) return "";
  const tzoff = new Date().getTimezoneOffset() * 60000;
  const strtime = new Date(n * 1000 - tzoff)
    .toISOString()
    .slice(0, 16)
    .replace("T", " ");
  return strtime;
}
