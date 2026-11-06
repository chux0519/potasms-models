"""Build the same frozen CPU encoder for macOS, iPhone and iPhone Simulator."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,shlex

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--developer-dir',type=Path);p.add_argument('--host-linker',type=Path);p.add_argument('--offline',action='store_true');a=p.parse_args();source=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
developer=(a.developer_dir or Path(subprocess.check_output(['xcode-select','-p'],text=True).strip())).resolve();tools=developer/'Toolchains/XcodeDefault.xctoolchain/usr/bin';headers=out/'include';headers.mkdir();shutil.copy2(source/'src/semantic_embed.h',headers/'semantic_embed.h');(headers/'module.modulemap').write_text('module POTASMSEncoder { header "semantic_embed.h" export * }\n');archives=[]
host_linker=None
if a.host_linker:
 host_linker=out/'host-linker.sh';compiler=a.host_linker.resolve();host_sdk=compiler.parents[4]/'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
 if not host_sdk.is_dir():raise ValueError('Host linker must come from a complete Xcode toolchain')
 host_linker.write_text('#!/bin/sh\nexec '+shlex.quote(str(compiler))+' "$@" -isysroot '+shlex.quote(str(host_sdk))+'\n');host_linker.chmod(0o755)
def command(args,log,cwd=None,env=None):
 with log.open('w') as f:subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for name,triple,platform,sdk_name,cpu in [('macos','aarch64-apple-darwin','MacOSX','MacOSX','apple-m1'),('iphone','aarch64-apple-ios','iPhoneOS','iPhoneOS','apple-a11'),('simulator','aarch64-apple-ios-sim','iPhoneSimulator','iPhoneSimulator','apple-m1')]:
 target=out/name;target.mkdir();sdk=developer/f'Platforms/{platform}.platform/Developer/SDKs/{sdk_name}.sdk';env=dict(os.environ,DEVELOPER_DIR=str(developer),SDKROOT=str(sdk),CARGO_TARGET_DIR=str(target/'cargo'))
 # Some new Apple linkers emit host proc-macro dylibs rejected by older dyld.
 # Keep the target SDK explicit while permitting a known-good host linker.
 if host_linker:env['CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER']=str(host_linker)
 if name=='macos':env['MACOSX_DEPLOYMENT_TARGET']='14.0'
 else:env['IPHONEOS_DEPLOYMENT_TARGET']='16.0';env['CFLAGS_'+triple.replace('-','_')]='-mios-simulator-version-min=16.0' if name=='simulator' else '-miphoneos-version-min=16.0';env['CXXFLAGS_'+triple.replace('-','_')]=env['CFLAGS_'+triple.replace('-','_')]
 command(['rustup','target','add',triple],target/'rust-target.log',env=env)
 command(['cargo','build','--locked','--release','--target',triple,*(['--offline'] if a.offline else [])],target/'rust-build.log',cwd=source/'ffi-source',env=env)
 rust=target/f'cargo/{triple}/release/libtokenizers_c.a'
 (target/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.22)
project(POTASMSAppleSDK C CXX)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("{source/'ggml'}" ggml)
add_library(pota_sdk STATIC "{source/'src/base_model.cpp'}" "{source/'src/gte.cpp'}" "{source/'src/jina_bert.cpp'}" "{source/'src/bert.cpp'}" "{source/'src/tokenizer.cpp'}" "{source/'src/utils.cpp'}" "{source/'src/factory.cpp'}" "{source/'src/student_embed.cpp'}" "{source/'hf_tokenizers/src/huggingface_tokenizer.cc'}")
target_include_directories(pota_sdk PUBLIC "{source/'src'}" "{source/'ggml/include'}" "{source/'hf_tokenizers/include'}" "{source/'third_party'}")
target_compile_options(pota_sdk PRIVATE -ffp-contract=off)
target_link_libraries(pota_sdk PUBLIC ggml "{rust}" "-framework Foundation" "-framework Security")
''')
 config=['cmake','-S',target,'-B',target/'cmake','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OSX_ARCHITECTURES=arm64','-DCMAKE_OSX_SYSROOT='+str(sdk),'-DCMAKE_OSX_DEPLOYMENT_TARGET='+('14.0' if name=='macos' else '16.0'),'-DCMAKE_C_COMPILER='+str(tools/'clang'),'-DCMAKE_CXX_COMPILER='+str(tools/'clang++'),'-DCMAKE_C_FLAGS=-mcpu='+cpu,'-DCMAKE_CXX_FLAGS=-mcpu='+cpu]
 if name!='macos':config.append('-DCMAKE_SYSTEM_NAME=iOS')
 command(config,target/'configure.log',env=env);command(['cmake','--build',target/'cmake','--target','pota_sdk','--parallel','4'],target/'cpp-build.log',env=env)
 library=target/'libPOTASMSEncoder.a';command([tools/'libtool','-static','-o',library,*sorted((target/'cmake').rglob('*.a')),rust],target/'combine.log',env=env);archives.append(library);print('APPLE_ENCODER_BUILT',name,library.stat().st_size,flush=True)
command([developer/'usr/bin/xcodebuild','-create-xcframework',*[arg for lib in archives for arg in ['-library',lib,'-headers',headers]],'-output',out/'POTASMSEncoder.xcframework'],out/'xcframework.log',env=dict(os.environ,DEVELOPER_DIR=str(developer)))
files={str(p.relative_to(out/'POTASMSEncoder.xcframework')):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (out/'POTASMSEncoder.xcframework').rglob('*') if p.is_file()};(out/'manifest.json').write_text(json.dumps({'source_provenance_sha256':hashlib.sha256((source/'PROVENANCE.json').read_bytes()).hexdigest(),'cpu_only':True,'python_dependency':False,'minimum_ios':'16.0','minimum_macos':'14.0','files':files},indent=2)+'\n');print('APPLE_XCFRAMEWORK_COMPLETE',flush=True)
