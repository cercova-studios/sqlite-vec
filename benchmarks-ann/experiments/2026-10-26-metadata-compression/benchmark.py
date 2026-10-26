import json,math,platform,random,sqlite3,statistics,struct,time,zlib,subprocess
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
LIB=str(ROOT/'dist/vec0.dylib')
def samples(fn):
    for _ in range(5): fn()
    batches=[]
    for _ in range(9):
        start=time.perf_counter_ns()
        for _ in range(20): fn()
        batches.append((time.perf_counter_ns()-start)/20/1000)
    return {'median_us':statistics.median(batches),'min_us':min(batches),'max_us':max(batches),'batches_us':batches}
def values(kind,n):
    rng=random.Random(42)
    if kind=='integer_categories': return 'integer',[rng.randrange(16) for _ in range(n)]
    if kind=='integer_sequence': return 'integer',[1700000000+i for i in range(n)]
    if kind=='integer_random64': return 'integer',[rng.getrandbits(64)-(1<<63) for _ in range(n)]
    if kind=='float_random': return 'float',[rng.uniform(-1e6,1e6) for _ in range(n)]
    if kind=='float_constant': return 'float',[1.23456789]*n
    if kind=='boolean_balanced': return 'boolean',[rng.randrange(2) for _ in range(n)]
    if kind=='boolean_sparse': return 'boolean',[int(rng.random()<.01) for _ in range(n)]
    if kind=='text_categories': return 'text',[f'tag{rng.randrange(16):05d}' for _ in range(n)]
    if kind=='text_random12': return 'text',[''.join(rng.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-',k=12)) for _ in range(n)]
    if kind=='text_random32': return 'text',[''.join(rng.choices('0123456789abcdef',k=32)) for _ in range(n)]
    raise ValueError(kind)
profiles=['integer_categories','integer_sequence','integer_random64','float_random','float_constant','boolean_balanced','boolean_sparse','text_categories','text_random12','text_random32']
receipts=[]
for n in [128,1024,4096]:
    for kind in profiles:
        sqltype,data=values(kind,n)
        db=sqlite3.connect(':memory:');db.enable_load_extension(True);db.load_extension(LIB)
        db.execute(f'create virtual table t using vec0(v float[2],m {sqltype},chunk_size={n})')
        vector=struct.pack('2f',0,0)
        db.executemany('insert into t(rowid,v,m) values (?,?,?)',[(i,vector,value) for i,value in enumerate(data)])
        db.commit()
        assert [r[0] for r in db.execute('select m from t order by rowid')]==data
        blobs=[r[0] for r in db.execute('select data from t_metadatachunks00')]
        assert len(blobs)==1
        raw=blobs[0]
        overflow=db.execute('select coalesce(sum(length(cast(data as blob))),0) from t_metadatatext00').fetchone()[0] if sqltype=='text' else 0
        result={'profile':kind,'chunk_rows':n,'raw_bytes':len(raw),'overflow_text_bytes':overflow,'frame_assumption_bytes':16,'codecs':{}}
        try:
            result['metadata_table_page_bytes']=db.execute("select sum(pgsize) from dbstat where name='t_metadatachunks00'").fetchone()[0]
        except sqlite3.Error: result['metadata_table_page_bytes']=None
        for level in [1,6]:
            compressed=zlib.compress(raw,level)
            assert zlib.decompress(compressed)==raw
            result['codecs'][str(level)]={'compressed_bytes':len(compressed),'framed_bytes':len(compressed)+16,'chosen_payload_bytes':min(len(raw),len(compressed)+16),'encode':samples(lambda:zlib.compress(raw,level)),'decode':samples(lambda:zlib.decompress(compressed))}
        if sqltype=='integer':
            width=(max(data)-min(data)).bit_length()
            result['integer_frame_of_reference_estimate']={'bits':width,'bytes_with_16_byte_header_8_byte_base_1_byte_width':25+math.ceil(n*width/8)}
        receipts.append(result);db.close()
meta={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'sqlite':sqlite3.sqlite_version,'zlib':zlib.ZLIB_RUNTIME_VERSION,'timing':'9 batches of 20 calls after 5 warmups; CPU codec microbenchmarks, not SQL timings','sqlite_pages':'dbstat bytes for the raw metadata shadow table; excludes other tables/indexes','scope':'synthetic full chunks from the unchanged vec0 engine; byte-exact roundtrips; no production codec or format change'}
args.output.write_text(json.dumps({'metadata':meta,'results':receipts},indent=2))
print(json.dumps(meta),flush=True)
for result in receipts:
    if result['chunk_rows']!=1024:continue
    c=result['codecs']['1']
    print(json.dumps({'profile':result['profile'],'raw_bytes':result['raw_bytes'],'overflow':result['overflow_text_bytes'],'framed_zlib1':c['framed_bytes'],'raw_fallback':c['chosen_payload_bytes'],'encode_median_us':round(c['encode']['median_us'],2),'decode_median_us':round(c['decode']['median_us'],2),'for_estimate':result.get('integer_frame_of_reference_estimate')}),flush=True)
print('All 30 stored-blob profiles and 60 codec roundtrips verified.',flush=True)
