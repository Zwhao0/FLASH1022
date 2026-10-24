"""Isolated node entry. No GPU launch unless explicitly invoked by the GPU owner."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .runtime import atomic

PKG='results.layer_campaign.stage_forward_v1.monitor_gc_268_sparse3_draft'


class Supervisor:
    def __init__(self,output,node,original_publish,kill=os.kill,clock=time.monotonic,sleep=time.sleep):
        self.output,self.node,self.original_publish=Path(output),node,original_publish
        self.kill,self.clock,self.sleep=kill,clock,sleep
        self.controllers=[];self.publishers=[];self.sequence=0;self.mode='bare';self.paused=False
    def _state(self,process):
        path=Path(f'/proc/{process.pid}/status')
        if path.exists():
            return next(line.split()[1] for line in path.read_text().splitlines() if line.startswith('State:'))
        raise RuntimeError('cannot attest owned controller process state')
    def _signal(self,paused):
        for process in self.controllers:
            if process.poll() is not None:raise RuntimeError('owned controller exited before pause/resume')
            self.kill(process.pid,signal.SIGSTOP if paused else signal.SIGCONT)
            deadline=self.clock()+5
            while (self._state(process)=='T')!=paused:
                if self.clock()>deadline:raise TimeoutError('owned controller pause/resume unverified')
                self.sleep(.005)
        self.paused=paused
    def resume_all(self):
        for process in self.controllers:
            try:self.kill(process.pid,signal.SIGCONT)
            except ProcessLookupError:pass
        self.paused=False
    def _flush(self):
        for publisher in self.publishers:self.original_publish(publisher)
        if any(any(p.partial.values()) for p in self.publishers):raise RuntimeError('partial publisher row at boundary')
    def _drained(self,physical):
        self._flush()
        if self.node!=0 or physical<=1:
            return dict(local_partial_rows=0,publisher_objects=len(self.publishers))
        if len(self.publishers)!=2:raise RuntimeError('both real publishers required')
        client=self.publishers[0].client
        deadline=self.clock()+15
        while True:
            self._flush()
            pending=client.get('online_pending');processed=client.get('online_processed_iteration')
            pending=int(pending or 0);processed=int(processed or 0)
            if pending==0 and processed>=physical-1:
                from flash_v1.monitor_channel import MonitorChannel,_HEADER
                channel=MonitorChannel(self.output/'monitor.bin')
                header=_HEADER.unpack_from(channel.memory);channel.close()
                if header[1]!=header[2]:raise RuntimeError('unconsumed monitor samples at boundary')
                return dict(online_pending=pending,processed_physical=processed,written=header[1],consumed=header[2],request_epoch=header[3])
            if self.clock()>deadline:raise TimeoutError('controller/publisher queues did not drain')
            self.sleep(.01)
    def poll(self):
        path=self.output/f'cost_mode_request_node{self.node}.json'
        if not path.exists():return
        request=json.loads(path.read_text())
        if request['sequence']<=self.sequence:return
        mode=request['mode'];physical=request['physical'];start=time.time();queue={}
        try:
            if mode=='bare':
                if self.mode!='bare':queue=self._drained(physical)
                self._signal(True)
            elif mode=='full':
                if self.mode=='bare':
                    if physical!=77:raise RuntimeError('first full block must start physical77')
                    if self.node==0:
                        from flash_v1.monitor_channel import MonitorChannel,_HEADER
                        channel=MonitorChannel(self.output/'monitor.bin');header=_HEADER.unpack_from(channel.memory);channel.close()
                        if header[1:4]!=(0,0,0):raise RuntimeError('nonempty monitor prefix before firstB')
                        queue=dict(written=0,consumed=0,request_epoch=0)
                    self._signal(False)
                else:queue=self._drained(physical)
            elif mode in ('done','abort'):
                self.resume_all()
            else:raise ValueError('invalid benchmark actor mode')
            self.mode=mode;self.sequence=request['sequence']
            row=dict(status='ready',**request,node=self.node,ack_walltime=time.time(),started_walltime=start,
                     controller_paused=self.paused,publisher_idle=mode=='bare',
                     controller_pids=[p.pid for p in self.controllers],queues=queue,
                     redis='shared common infrastructure left running; no A FLASH commands')
        except BaseException as error:
            self.resume_all();row=dict(status='failed',**request,error=repr(error),node=self.node,ack_walltime=time.time())
            atomic(self.output/f'cost_mode_ack_node{self.node}.json',row)
            raise
        atomic(self.output/f'cost_mode_ack_node{self.node}.json',row)
        with (self.output/f'cost_actor_ack_history_node{self.node}.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
    def publish(self,publisher):
        if publisher not in self.publishers:self.publishers.append(publisher)
        self.poll()
        if self.mode=='bare':return 0
        return self.original_publish(publisher)


def main():
    from flash_v1.scripts import v2_transport
    from . import node_collection_cost as node_collection
    plan=Path(sys.argv[sys.argv.index('--plan')+1]);spec=json.loads(plan.read_text())
    if not spec.get('monitor_sameprocess_cost') or spec.get('steps')!=268 or spec.get('max_profiles')!=3:
        raise ValueError('explicit reviewed sameprocess cost contract required')
    if not spec.get('cost_harness_files') or any(hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected
            for name,expected in spec['cost_harness_files'].items()):
        raise ValueError('isolated cost source deployment differs from reviewed freeze')
    output=Path('results/data16_sessions')/spec['session_id'];node=int(sys.argv[sys.argv.index('--node-rank')+1])
    original_popen=subprocess.Popen;original_publish=v2_transport.EvidencePublisher.publish
    supervisor=Supervisor(output,node,original_publish)
    def popen(command,*args,**kwargs):
        command=list(command) if isinstance(command,list) else command
        if isinstance(command,list):
            if 'flash_v1.online_v2' in command:command[command.index('flash_v1.online_v2')]=PKG+'.controller_cost'
            if 'flash_v1.train_pipeline' in command:
                command[command.index('flash_v1.train_pipeline')]=PKG+'.train_pipeline_cost'
                # All actors are owned direct children already created by the
                # standard supervisor. Pause controller before training starts.
                supervisor._signal(True)
        process=original_popen(command,*args,**kwargs)
        if isinstance(command,list) and PKG+'.controller_cost' in command:supervisor.controllers.append(process)
        if isinstance(command,list) and PKG+'.train_pipeline_cost' in command:
            # Mode control is independent of samples AND publisher activity.
            # This is the existing common supervision poll, no new actor/thread.
            original_poll=process.poll
            def training_poll():
                supervisor.poll()
                return original_poll()
            process.poll=training_poll
        if output.exists():
            atomic(output/f'cost_entry_node{node}.json',dict(
                entry=PKG+'.node_cost',training=PKG+'.train_pipeline_cost',controller=PKG+'.controller_cost',
                plan_file_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                controller_pids=[p.pid for p in supervisor.controllers],redis_not_stopped=True))
        return process
    def publish(publisher):return supervisor.publish(publisher)
    subprocess.Popen=popen;v2_transport.EvidencePublisher.publish=publish
    try:node_collection.main()
    finally:
        supervisor.resume_all()
        subprocess.Popen=original_popen;v2_transport.EvidencePublisher.publish=original_publish


if __name__=='__main__':main()
