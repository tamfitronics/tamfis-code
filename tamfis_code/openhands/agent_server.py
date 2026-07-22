from __future__ import annotations
import argparse, asyncio, os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import uvicorn

from .conversation import ConversationManager, ConversationState
from .agent import TamfisAgent

STATE_ROOT=Path(os.environ.get("TAMFIS_CODE_SERVER_STATE", Path.home()/".local/share/tamfis-code/agent-server"))
manager=ConversationManager(STATE_ROOT)
app=FastAPI(title="Tamfis-Code Agent Server",version="0.6.0")

class CreateConversation(BaseModel):
    workspace:str
    metadata:dict[str,Any]=Field(default_factory=dict)
class MessageRequest(BaseModel):
    content:str
class RunRequest(BaseModel):
    objective:str
    provider:str="auto"
    model:str|None=None
    approval_policy:str="ask"
    read_only:bool=False
    max_rounds:int=30

class ToolRequest(BaseModel):
    name:str
    arguments:dict[str,Any]=Field(default_factory=dict)
    approval_policy:str="ask"

@app.get("/health")
def health():return {"ok":True,"service":"tamfis-code-agent-server","version":"0.6.0"}
@app.post("/v1/conversations")
def create(req:CreateConversation):
    try: conv=manager.create(req.workspace,metadata=req.metadata)
    except Exception as exc: raise HTTPException(400,str(exc))
    return {"id":conv.id,"state":conv.state.value,"workspace":str(conv.workspace.root)}
@app.get("/v1/conversations")
def list_conversations():return manager.list()
@app.get("/v1/conversations/{cid}/events")
def events(cid:str,after:int=0):return [e.as_dict() for e in manager.events.read(cid,after=after)]
@app.post("/v1/conversations/{cid}/messages")
def message(cid:str,req:MessageRequest):
    try: event=manager.get(cid).send_message(req.content)
    except KeyError: raise HTTPException(404,"conversation not found")
    return event.as_dict()

@app.post("/v1/conversations/{cid}/run")
async def run_agent(cid:str,req:RunRequest):
    try:
        conversation=manager.get(cid)
    except KeyError:
        raise HTTPException(404,"conversation not found")
    agent=TamfisAgent(conversation)
    result=await agent.run(req.objective,provider=req.provider,model=req.model,approval_policy=req.approval_policy,read_only=req.read_only,max_rounds=req.max_rounds)
    return {"status":result.status,"summary":result.summary,"error":result.error}

@app.post("/v1/conversations/{cid}/tools")
async def tool(cid:str,req:ToolRequest):
    try:return await manager.get(cid).invoke_tool(req.name,req.arguments,approval_policy=req.approval_policy)
    except KeyError:raise HTTPException(404,"conversation not found")
@app.post("/v1/conversations/{cid}/pause")
def pause(cid:str):manager.get(cid).pause();return {"state":"paused"}
@app.post("/v1/conversations/{cid}/resume")
def resume(cid:str):manager.get(cid).resume();return {"state":"running"}
@app.post("/v1/conversations/{cid}/cancel")
def cancel(cid:str):manager.get(cid).cancel();return {"state":"cancelled"}
@app.get("/v1/conversations/{cid}/workspace/files")
def files(cid:str,path:str="."):return manager.get(cid).workspace.list_files(path)
@app.post("/v1/conversations/{cid}/workspace/snapshot")
def snapshot(cid:str,label:str="checkpoint"):return manager.get(cid).workspace.snapshot(label=label)
@app.websocket("/v1/conversations/{cid}/stream")
async def stream(websocket:WebSocket,cid:str,after:int=0):
    await websocket.accept();sequence=after
    try:
        while True:
            events=manager.events.read(cid,after=sequence)
            for event in events:
                await websocket.send_json(event.as_dict());sequence=event.sequence
            await asyncio.sleep(.2)
    except WebSocketDisconnect:return

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--host",default="127.0.0.1");parser.add_argument("--port",type=int,default=9600);parser.add_argument("--workers",type=int,default=1);args=parser.parse_args();uvicorn.run("tamfis_code.openhands.agent_server:app",host=args.host,port=args.port,workers=args.workers)
