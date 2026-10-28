"""Prepare full BEIR SciFact and pinned, locally generated BGE embeddings."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import platform
import time
import urllib.request
import zipfile
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL = 'BAAI/bge-small-en-v1.5'
REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
DATA_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
DATA_SHA = '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165'
INSTRUCTION = 'Represent this sentence for searching relevant passages: '


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def download(url, path):
    if not path.exists():
        partial = path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,partial)
        partial.replace(path)


def encode(texts, tokenizer, session):
    encoded = tokenizer.encode_batch(texts)
    vectors = np.empty((len(texts),384),dtype=np.float32)
    order = sorted(range(len(texts)),key=lambda i:len(encoded[i].ids))
    start = time.monotonic()
    for offset in range(0,len(order),16):
        indices = order[offset:offset+16]
        width = max(len(encoded[i].ids) for i in indices)
        inputs = {name:np.zeros((len(indices),width),dtype=np.int64)
                  for name in ('input_ids','attention_mask','token_type_ids')}
        for row,index in enumerate(indices):
            item = encoded[index];n=len(item.ids)
            inputs['input_ids'][row,:n] = item.ids
            inputs['attention_mask'][row,:n] = item.attention_mask
            inputs['token_type_ids'][row,:n] = item.type_ids
        hidden = session.run(['last_hidden_state'],inputs)[0]
        cls = hidden[:,0,:]
        norms = np.sqrt(np.einsum('ij,ij->i',cls,cls))
        assert np.isfinite(cls).all() and (norms>0).all()
        vectors[indices] = cls/norms[:,None]
        if offset % 512 == 0:
            print(f'Embedded {min(offset+16,len(order))}/{len(order)} in {time.monotonic()-start:.1f}s',flush=True)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.sqrt(np.einsum('ij,ij->i',vectors,vectors)),1,atol=1e-6)
    return vectors, sum(len(item.ids)==512 for item in encoded)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'.work')
    args=parser.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    download(DATA_URL,work/'scifact.zip')
    assert digest(work/'scifact.zip')==DATA_SHA
    urls={}
    for remote,local in [('onnx/model.onnx','model.onnx'),('tokenizer.json','model-tokenizer.json')]:
        url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{remote}'
        download(url,work/local);urls[local]=url
    with zipfile.ZipFile(work/'scifact.zip') as archive:
        documents=[json.loads(line) for line in archive.read('scifact/corpus.jsonl').splitlines()]
        queries={r['_id']:r for r in (json.loads(line) for line in archive.read('scifact/queries.jsonl').splitlines())}
        qrels={}
        for row in csv.DictReader(io.StringIO(archive.read('scifact/qrels/test.tsv').decode()),delimiter='\t'):
            qrels.setdefault(row['query-id'],{})[row['corpus-id']]=int(row['score'])
    documents.sort(key=lambda r:int(r['_id']));query_ids=sorted(qrels,key=int)
    assert len(documents)==5183 and len(query_ids)==300 and sum(map(len,qrels.values()))==339
    assert len({d['_id'] for d in documents})==5183
    assert all(set(judgments)<=set(d['_id'] for d in documents) for judgments in qrels.values())
    tokenizer=Tokenizer.from_file(str(work/'model-tokenizer.json'))
    tokenizer.enable_truncation(max_length=512)
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(work/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    start=time.monotonic()
    doc_vectors,doc_capped=encode([d['title']+' '+d['text'] for d in documents],tokenizer,session)
    query_vectors,query_capped=encode([INSTRUCTION+queries[q]['text'] for q in query_ids],tokenizer,session)
    # Padding must not change the embedding of a fixed text.
    sample=[documents[0]['title']+' '+documents[0]['text'],documents[-1]['title']+' '+documents[-1]['text']]
    combined,_=encode(sample,tokenizer,session)
    separate=np.vstack([encode([text],tokenizer,session)[0] for text in sample])
    assert np.allclose(combined,separate,atol=2e-6,rtol=2e-5)
    np.save(work/'documents.npy',doc_vectors);np.save(work/'queries.npy',query_vectors)
    (work/'data.json').write_text(json.dumps(dict(documents=documents,queries=[queries[q] for q in query_ids],qrels=qrels)))
    manifest=dict(dataset_url=DATA_URL,dataset_sha256=DATA_SHA,model=MODEL,revision=REVISION,
        urls=urls,dimensions=384,max_tokens=512,pooling='CLS then L2 normalization',query_instruction=INSTRUCTION,
        document_input='title + single space + text; no instruction',queries=300,documents=5183,positive_judgments=339,
        documents_at_token_cap=doc_capped,queries_at_token_cap=query_capped,padding_check=True,
        onnxruntime=ort.__version__,numpy=np.__version__,python=platform.python_version(),platform=platform.platform(),
        providers=session.get_providers(),intra_threads=4,inter_threads=1,batch_size=16,seconds=time.monotonic()-start,
        files={name:digest(work/name) for name in ['scifact.zip','model.onnx','model-tokenizer.json','documents.npy','queries.npy','data.json']})
    (work/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
