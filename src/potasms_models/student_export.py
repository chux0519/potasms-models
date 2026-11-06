"""BERT GGUF export keeps learned pooler as a separate exact FP32 native layer."""
from pathlib import Path
import importlib.util,json,os,shutil,struct,subprocess,types,argparse
from .student_prepare_model import RUN,sha,write
from .paths import DATA

def export(selection_path=None,folder_path=None):
 import torch
 from transformers import AutoModel,AutoTokenizer
 from safetensors.torch import load_file
 parent=DATA/'runtime-source-v1';selection=json.loads((selection_path or RUN/'fit-01/selection.json').read_text());checkpoint=Path(selection['checkpoint']);assert sha(checkpoint/'model.safetensors')==selection['weights_sha256'];folder=folder_path or RUN/'native-export-01';folder.mkdir(exist_ok=False)
 spec=importlib.util.spec_from_file_location('frozen_student_converter',parent/'scripts/convert.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.MODEL_CONFIG_MAP[str(checkpoint)]=mod.BertConversionConfig
 mod.load_auto_model=lambda path:AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,add_pooling_layer=False,torch_dtype=torch.float32,use_safetensors=True,attn_implementation='eager')
 mod.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path,**kw:AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False))
 state=load_file(str(checkpoint/'model.safetensors'));w=state['pooler.dense.weight'].numpy();b=state['pooler.dense.bias'].numpy();assert w.shape==(128,128) and b.shape==(128,);pooler=struct.pack('<4sI',b'S5P1',128)+w.astype('<f4').tobytes()+b.astype('<f4').tobytes()
 for precision in ['f32','f16']:mod.convert_hf(str(checkpoint),str(folder/f'encoder.{precision}.gguf'),precision)
 env=dict(os.environ);env.pop('EMBEDDINGS_CPP_SKIP_QUANT_PATTERNS',None);subprocess.run([str(DATA/'runtime-build/cmake/quantize'),str(folder/'encoder.f32.gguf'),str(folder/'encoder.q8_0.gguf'),'q8_0'],env=env,check=True)
 for p in folder.glob('*.gguf'):p.with_name(p.name+'.pooler.bin').write_bytes(pooler)
 for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json']:shutil.copy2(checkpoint/name,folder/name)
 report=dict(checkpoint_sha256=selection['weights_sha256'],converter_sha256=sha(parent/'scripts/convert.py'),quantizer_sha256=sha(DATA/'runtime-build/cmake/quantize'),pooler='FP32 learned dense+tanh before L2, separate .pooler.bin required',projection_exported=False,parameters=sum(t.numel() for t in state.values()),source_sha256=sha(__file__),files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir() if p.is_file()});write(folder/'manifest.json',report);print('STUDENT_NATIVE_EXPORT_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path);p.add_argument('--folder',type=Path);a=p.parse_args();export(a.selection,a.folder)
