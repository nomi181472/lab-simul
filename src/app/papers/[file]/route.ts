import { createReadStream, promises as fs } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { MANIFEST } from "@/labs/wifi-sensing/data/manifest";

/* Serves corpus PDFs to the in-app viewer.
 *
 * Two security properties, both deliberate:
 *  1. ALLOWLIST — a request is only served if its filename is exactly a
 *     `fileName` in the generated manifest. Nothing else on disk is reachable,
 *     regardless of what the path contains.
 *  2. TRAVERSAL GUARD — the resolved path is re-checked to be inside the corpus
 *     directory, so a crafted filename cannot escape even in principle.
 *
 * Range requests are supported because the browser's native PDF viewer issues
 * them; without this, large PDFs fail to load in Chrome.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CORPUS_DIR = path.join(process.cwd(), "papers", "wifi_sensing");
const ALLOWED = new Set(MANIFEST.map((p) => p.fileName));

const TYPES: Record<string, string> = {
  ".pdf": "application/pdf",
};

function contentType(fileName: string) {
  return TYPES[path.extname(fileName).toLowerCase()] ?? "application/octet-stream";
}

function notFound(msg: string) {
  return new Response(msg, { status: 404, headers: { "content-type": "text/plain; charset=utf-8" } });
}

export async function GET(request: Request, { params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;

  // the URL segment arrives percent-encoded; decode once and then require an
  // exact allowlist hit, which rejects any attempt at traversal by construction
  let name: string;
  try {
    name = decodeURIComponent(file);
  } catch {
    return notFound("bad request");
  }

  if (!ALLOWED.has(name)) return notFound("not in corpus manifest");

  const full = path.resolve(CORPUS_DIR, name);
  const root = path.resolve(CORPUS_DIR);
  if (full !== path.join(root, name)) return notFound("bad request");

  let stat: Awaited<ReturnType<typeof fs.stat>>;
  try {
    stat = await fs.stat(full);
  } catch {
    return notFound("file missing from corpus");
  }
  if (!stat.isFile()) return notFound("not a file");

  const total = stat.size;
  const type = contentType(name);
  const base: Record<string, string> = {
    "content-type": type,
    "accept-ranges": "bytes",
    "cache-control": "public, max-age=31536000, immutable",
    "content-disposition": `inline; filename="${name.replace(/"/g, "")}"`,
  };

  const range = request.headers.get("range");
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (m) {
      const rawStart = m[1];
      const rawEnd = m[2];
      let start: number;
      let end: number;
      if (rawStart === "" && rawEnd !== "") {
        // suffix range: last N bytes
        const n = Number(rawEnd);
        start = Math.max(total - n, 0);
        end = total - 1;
      } else {
        start = Number(rawStart || 0);
        end = rawEnd === "" ? total - 1 : Math.min(Number(rawEnd), total - 1);
      }
      if (Number.isFinite(start) && Number.isFinite(end) && start >= 0 && start <= end && start < total) {
        const stream = createReadStream(full, { start, end });
        return new Response(Readable.toWeb(stream) as ReadableStream, {
          status: 206,
          headers: {
            ...base,
            "content-range": `bytes ${start}-${end}/${total}`,
            "content-length": String(end - start + 1),
          },
        });
      }
      return new Response(null, {
        status: 416,
        headers: { "content-range": `bytes */${total}` },
      });
    }
  }

  const stream = createReadStream(full);
  return new Response(Readable.toWeb(stream) as ReadableStream, {
    status: 200,
    headers: { ...base, "content-length": String(total) },
  });
}

export async function HEAD(request: Request, { params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  let name: string;
  try {
    name = decodeURIComponent(file);
  } catch {
    return new Response(null, { status: 400 });
  }
  if (!ALLOWED.has(name)) return new Response(null, { status: 404 });
  const full = path.resolve(CORPUS_DIR, name);
  if (full !== path.join(path.resolve(CORPUS_DIR), name)) return new Response(null, { status: 400 });
  let stat: Awaited<ReturnType<typeof fs.stat>>;
  try {
    stat = await fs.stat(full);
  } catch {
    return new Response(null, { status: 404 });
  }
  return new Response(null, {
    status: 200,
    headers: {
      "content-type": contentType(name),
      "content-length": String(stat.size),
      "accept-ranges": "bytes",
    },
  });
}