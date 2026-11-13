#!/usr/bin/env python3
"""Voice2Action reproducible TTS study: PCM16 -> log-mel -> CNN -> safe virtual action.
NO ASR, NO physical actuator, NO verified CVNSS4 codec integration.
"""
import argparse
import json
import math
import platform
import random
import subprocess
import time
import wave
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly, stft

ROOT=Path(__file__).resolve().parent
FS=16000
ACTIONS=("stop","forward","left","right")
PHRASES=("dừng lại","tiến lên","rẽ trái","rẽ phải")
VOICES=("vi","vi-hue","vi-sgn")

def lexicon():
    return json.loads((ROOT/"lexicon.json").read_text(encoding="utf-8"))

def wav(path):
    with wave.open(str(path),"rb") as w:
        if w.getsampwidth()!=2: raise ValueError("WAV must be PCM16")
        sr=w.getframerate()
        n=w.getnchannels()
        a=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype("float32")/32768
    if n>1: a=a.reshape(-1,n).mean(axis=1)
    return a,sr

def mel():
    hm=lambda h:2595*np.log10(1+np.asarray(h)/700)
    mh=lambda m:700*(10**(np.asarray(m)/2595)-1)
    edges=mh(np.linspace(hm(60),hm(8000),34))
    f=np.linspace(0,8000,257)
    w=np.zeros((32,257),dtype="float32")
    for i in range(32):
        lo,mid,hi=edges[i:i+3]
        w[i]=np.maximum(0,np.minimum((f-lo)/(mid-lo),(hi-f)/(hi-mid)))
    return w

FILTER=mel()
def spec(audio,sr):
    audio=np.asarray(audio,dtype="float32")
    if sr<=0 or not np.isfinite(audio).all(): raise ValueError("invalid audio")
    if sr!=FS:
        g=math.gcd(FS,int(sr))
        audio=resample_poly(audio,FS//g,int(sr)//g).astype("float32")
    if len(audio)<3200 or np.sqrt(np.mean(audio**2))<.002: raise ValueError("silent or short")
    audio=audio[:28000]
    audio=np.pad(audio,(0,max(0,28000-len(audio))))
    _,_,z=stft(audio,fs=FS,window="hann",nperseg=512,noverlap=256,boundary=None,padded=False)
    return np.log1p(10000*(FILTER@(np.abs(z)**2))).astype("float32")

def generate(folder,variants=12):
    rng=random.Random(20261113);wavs=folder/"wavs";wavs.mkdir(parents=True,exist_ok=True)
    rows=[]
    for voice in VOICES:
        for action,phrase in zip(ACTIONS,PHRASES):
            for i in range(variants):
                speed=rng.randrange(132,202);pitch=rng.randrange(35,66)
                split="test" if voice=="vi-sgn" else ("val" if i>=int(.8*variants) else "train")
                p=wavs/(voice+"_"+action+"_"+str(i)+".wav")
                subprocess.run(["espeak","-v",voice,"-s",str(speed),"-p",str(pitch),
                                "-w",str(p),phrase],check=True,capture_output=True)
                rows.append(dict(audio=str(p.relative_to(folder)),split=split,action=action,voice=voice,
                                 speed=speed,pitch=pitch))
    (folder/"manifest.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    return {"samples":len(rows),"split":{s:sum(r["split"]==s for r in rows) for s in ("train","val","test")}}

def dataset(folder,split):
    rows=[json.loads(s) for s in (folder/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    selected=[r for r in rows if r["split"]==split]
    x=np.stack([spec(*wav(folder/r["audio"])) for r in selected])
    y=np.array([ACTIONS.index(r["action"]) for r in selected],dtype="int64")
    return x,y

def model():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(1,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(768,48),
        nn.ReLU(),nn.Linear(48,4))

def train(folder,checkpoint,epochs=25):
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261113);np.random.seed(20261113)
    x,y=dataset(folder,"train");v,vy=dataset(folder,"val")
    mean=float(x.mean());sd=float(x.std()+1e-5)
    x=torch.from_numpy(((x-mean)/sd)[:,None]);y=torch.from_numpy(y)
    v=torch.from_numpy(((v-mean)/sd)[:,None]);vy=torch.from_numpy(vy)
    net=model();opt=torch.optim.Adam(net.parameters(),lr=.002)
    best=-1;weights=None
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(len(x)).split(16):
            loss=torch.nn.functional.cross_entropy(net(x[ids]),y[ids])
            opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():acc=float((net(v).argmax(1)==vy).float().mean())
        if acc>=best:
            best=acc;weights={k:t.detach().clone() for k,t in net.state_dict().items()}
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state":weights,"mean":mean,"std":sd,"labels":ACTIONS},checkpoint)
    return {"val_accuracy":best,"train_n":len(x),"val_n":len(v)}

def load(checkpoint):
    import torch
    d=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if tuple(d["labels"])!=ACTIONS:raise ValueError("label mismatch")
    n=model();n.load_state_dict(d["state"]);n.eval()
    return n,float(d["mean"]),float(d["std"])

def classify(net,mean,std,s):
    import torch
    with torch.no_grad():
        p=torch.softmax(net(torch.from_numpy(((s-mean)/std)[None,None])),dim=1)[0].numpy()
    inds=p.argsort()
    return ACTIONS[int(inds[-1])],float(p[inds[-1]]),float(p[inds[-1]]-p[inds[-2]])

def safety(action,confidence,margin,state=None):
    """Deterministic allowlist/kinematics gate. It cannot detect all wrong allowed actions."""
    if state is None:state={"x":0.0,"y":0.0,"heading":0,"estop":False}
    if state.get("estop"):return {"decision":"REJECT","reason":"estop"}
    lookup=lexicon()["actions"]
    if action not in lookup:return {"decision":"REJECT","reason":"not_in_lexicon"}
    if not np.isfinite(confidence) or not np.isfinite(margin) or confidence<.65 or margin<.15:
        return {"decision":"REJECT","reason":"low_confidence"}
    task=lookup[action];kind=task["type"];value=task["parameter"]
    if not isinstance(value,(int,float)) or not math.isfinite(value):
        return {"decision":"REJECT","reason":"invalid_parameter"}
    new=state.copy()
    if kind=="stop" and value==0:return {"decision":"ALLOW","next":new}
    if kind=="turn" and abs(value)<=90:
        new["heading"]=(new["heading"]+value)%360
        return {"decision":"ALLOW","next":new}
    if kind=="move" and 0<value<=.5:
        theta=math.radians(new["heading"]);nx=new["x"]+value*math.cos(theta);ny=new["y"]+value*math.sin(theta)
        if abs(nx)>1 or abs(ny)>1:return {"decision":"REJECT","reason":"boundary"}
        ax,ay=new["x"],new["y"];dx,dy=nx-ax,ny-ay
        u=max(0,min(1,((.6-ax)*dx+(.5-ay)*dy)/max(dx*dx+dy*dy,1e-9)))
        if math.hypot(ax+u*dx-.6,ay+u*dy-.5)<.2:
            return {"decision":"REJECT","reason":"keepout"}
        new.update(x=nx,y=ny);return {"decision":"ALLOW","next":new}
    return {"decision":"REJECT","reason":"parameter_not_allowed"}

def evaluate(folder,checkpoint,report):
    import torch
    torch.set_num_threads(2)
    net,mean,std=load(checkpoint);x,y=dataset(folder,"test")
    matrix=np.zeros((4,4),dtype=int);times=[];accepted=0;good_allowed=0
    for s,truth in zip(x,y):
        t=time.perf_counter()
        pred,c,m=classify(net,mean,std,s);response=safety(pred,c,m)
        times.append((time.perf_counter()-t)*1000)
        matrix[truth,ACTIONS.index(pred)]+=1
        accepted+=response["decision"]=="ALLOW"
        good_allowed+=response["decision"]=="ALLOW" and ACTIONS[truth]==pred
    # Includes preprocessing+network+safety; excludes audio acquisition and robot actuation.
    rows=[json.loads(r) for r in (folder/"manifest.jsonl").read_text().splitlines()]
    a,sr=wav(folder/next(r["audio"] for r in rows if r["split"]=="test"))
    full=[]
    for _ in range(50):
        t=time.perf_counter()
        p=spec(a,sr);pred,c,m=classify(net,mean,std,p);safety(pred,c,m)
        full.append((time.perf_counter()-t)*1000)
    data=dict(test_source="eSpeak TTS held-out profile vi-sgn; NOT human speech",n_test=len(y),
        accuracy=float(np.trace(matrix)/len(y)),allow_rate=accepted/len(y),
        correct_and_allowed=good_allowed/len(y),labels=ACTIONS,confusion=matrix.tolist(),
        inference_ms_p50=float(np.median(times)),full_compute_ms_p50=float(np.median(full)),
        full_compute_ms_p95=float(np.percentile(full,95)),
        excludes=["capture","VAD","network","actuation"],
        cvnss4_codec_integrated=False,hardware_validated=False,
        runtime={"python":platform.python_version(),"torch":torch.__version__})
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    return data

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="cmd",required=True)
    for name in ("generate","train","eval","infer"):
        p=sub.add_parser(name)
        p.add_argument("--data",type=Path,default=ROOT/"data")
        p.add_argument("--model",type=Path,default=ROOT/"output"/"cnn.pt")
        if name=="train":p.add_argument("--epochs",type=int,default=25)
        if name=="generate":p.add_argument("--variants",type=int,default=12)
        if name=="eval":p.add_argument("--report",type=Path,default=ROOT/"output"/"report.json")
        if name=="infer":p.add_argument("wav",type=Path)
    opt=parser.parse_args()
    if opt.cmd=="generate":out=generate(opt.data,opt.variants)
    elif opt.cmd=="train":out=train(opt.data,opt.model,opt.epochs)
    elif opt.cmd=="eval":out=evaluate(opt.data,opt.model,opt.report)
    else:
        net,mean,std=load(opt.model);pred,c,m=classify(net,mean,std,spec(*wav(opt.wav)))
        out={"action":pred,"confidence":c,"margin":m,"gate":safety(pred,c,m)}
    print(json.dumps(out,ensure_ascii=False,indent=2))
