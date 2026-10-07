#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures as cf, json, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'papers' / 'networking'
CACHE = ROOT / '.cache' / 'harvest_networking'

TARGET = 300
PDF_TIMEOUT = 120
ARXIV_PAUSE = 6.0
DOWNLOAD_WORKERS = 2
MAX_RETRIES = 3
UA = {'User-Agent': 'paper-harvest/1.0 (mailto:nomansoomro51@gmail.com)'}
ARXIV_UA = UA

SIGNAL = re.compile(r'network|protocol|routing|congestion|tcp|quic|sdn|nfv|overlay|p2p|cdn|content delivery|datacenter|wireless|mobile', re.I)
TASK = re.compile(r'design|architecture|performance|protocol|scalab|latenc|throughput|optimi|reliab|efficien', re.I)
OFFTOPIC = re.compile(r'social network analysis|blockchain|cryptograph|deep learning for network ml|ai for network', re.I)

ARXIV_QUERIES = ['cat:cs.NI', 'cat:cs.DC']

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def cached_get(url, headers, cache_key, timeout=60, tag=''):
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / cache_key
    if c.exists() and c.stat().st_size > 0: return c.read_bytes()
    for attempt in range(1, MAX_RETRIES+1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            c.write_bytes(data)
            return data
        except Exception as e:
            if attempt == MAX_RETRIES: raise
            pause = 2**attempt
            log(f'retry {attempt}/{MAX_RETRIES} {tag}: {e}; sleep {pause}s')
            time.sleep(pause)
    return b''

def unescape(s):
    from html import unescape as hu
    return hu(s)

def arxiv_query(q, start=0, max_results=100):
    params = urllib.parse.urlencode({'search_query': q, 'start': start, 'max_results': max_results, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
    url = f'http://export.arxiv.org/api/query?{params}'
    data = cached_get(url, ARXIV_UA, f'arxiv_{abs(hash(q+str(start)))}.xml', timeout=90, tag='arxiv')
    text = data.decode('utf-8','replace')
    entries=[]
    for m in re.finditer(r'<entry>(.*?)</entry>', text, re.S):
        e = m.group(1)
        def g(tag):
            mm = re.search(rf'<{tag}>(.*?)</{tag}>', e, re.S)
            if not mm: return ''
            return re.sub(r'\s+',' ', unescape(mm.group(1).strip()))
        idm = re.search(r'<id>.*?/(\d+\.\d+)</id>', e)
        arxiv = idm.group(1) if idm else g('id').split('/')[-1]
        updated = g('updated')[:10] or g('published')[:10]
        year = int(updated[:4]) if updated[:4].isdigit() else 0
        authors=[]
        for am in re.finditer(r'<author><name>(.*?)</name>', e, re.S):
            authors.append(unescape(am.group(1).strip()))
        title = g('title'); absr = g('summary')
        doi = ''
        mdoi = re.search(r'doi:\s*(10\.\d{4,}/[^\s<]+)', absr, re.I)
        if mdoi:
            doi = mdoi.group(1).rstrip('.,)')
        entries.append({'arxiv':arxiv,'doi':doi,'title':title,'year':year,'venue':'','authors':authors,'citations':0,'pdfUrl':f'https://arxiv.org/pdf/{arxiv}.pdf','source':'arxiv','abstract':absr})
    return entries

def relevant(c):
    t = (c.get('title') or '') + ' ' + (c.get('abstract') or '')
    if not t: return False
    if OFFTOPIC.search(t): return False
    return bool(SIGNAL.search(t) and TASK.search(t))

def arxiv_gate():
    seen=set(); out=[]
    for q in ARXIV_QUERIES:
        for start in (0,100,200,300,400,500):
            try:
                batch = arxiv_query(q,start=start,max_results=100)
            except Exception as e:
                log(f'arxiv err {start}: {e}')
                break
            if not batch: break
            for c in batch:
                if c['year'] and c['year'] <= 2010: continue
                if c['arxiv'] in seen: continue
                seen.add(c['arxiv'])
                if relevant(c): out.append(c)
            time.sleep(ARXIV_PAUSE)
            if len(batch) < 100: break
    log(f'arxiv candidates: {len(out)}')
    return out
def valid_pdf(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 5000: return False
        with open(path,'rb') as f: h=f.read(4)
        return h==b'%PDF'
    except: return False

def fetch(url, dest):
    if not url: return False,'no-url'
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=PDF_TIMEOUT) as r:
            data = r.read()
        dest.write_bytes(data)
        return True,f'{len(data)}B'
    except Exception as e:
        return False,f'{type(e).__name__}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=TARGET)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cands = arxiv_gate()
    cands.sort(key=lambda c:(-(c.get('year') or 0), c['title']))
    manifest=[]; failed=[]
    def job(c):
        dest = OUT / f"{c['arxiv']}.pdf"
        if valid_pdf(dest): return c
        time.sleep(ARXIV_PAUSE / max(1,DOWNLOAD_WORKERS))
        ok,why = fetch(c['pdfUrl'], dest)
        if ok and valid_pdf(dest): return c
        dest.unlink(missing_ok=True); failed.append([c['arxiv'], why]); return None
    with cf.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        for n,r in enumerate(ex.map(job,cands),1):
            if r: manifest.append(r)
            if n%10==0: log(f'{len(manifest)}/{n}')
            if len(manifest)>=args.target: break
    manifest.sort(key=lambda c:(-(c.get('year') or 0), c['title']))
    rows=[]
    for i,c in enumerate(manifest,1):
        rows.append({'id':f'N{i:03d}','arxiv':c['arxiv'],'doi':c.get('doi',''),'title':c['title'],'year':c.get('year'),'venue':c.get('venue',''),'authors':c.get('authors',[]),'citations':c.get('citations',0),'fileName':f"{c['arxiv']}.pdf",'source':c.get('source','')})
    (OUT/'manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    (OUT/'download_summary.json').write_text(json.dumps({'success':len(rows),'failed':len(failed)},indent=2))
    (OUT/'unavailable.json').write_text(json.dumps(failed,indent=2))
    log(f'done {len(rows)}')
    return 0
if __name__=='__main__': sys.exit(main())
