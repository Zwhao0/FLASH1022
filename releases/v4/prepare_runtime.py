"""Install frozen model receipts for this clone without refitting or changing hashes."""
from pathlib import Path
import argparse, copy, hashlib, json

MODEL_REL=Path('results/v4/frozen/r11_kernel_match_v3')


def digest(path):
    data=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def install(root, package):
    original=json.loads((package/'freeze_receipt.json').read_text())
    model=package/'model.json'
    assert {k:digest(model)[k] for k in ('bytes','sha256')}=={k:original['model'][k] for k in ('bytes','sha256')},'model bytes changed'
    receipt=copy.deepcopy(original)
    mapping=[]
    for row in receipt['source_files']:
        # Frozen receipt binds exactly the six unchanged inference sources.
        relative=Path('flash_v4')/Path(row['path']).name
        current=digest(root/relative)
        assert all(current[k]==row[k] for k in ('bytes','sha256')),f'code hash mismatch: {relative}'
        mapping.append(dict(original=row['path'],relocated=current['path'],sha256=row['sha256']))
        row.update(current)
    target=root/MODEL_REL
    target.mkdir(parents=True,exist_ok=True)
    model_path=target/'model.json'
    if model_path.exists():
        assert model_path.read_bytes()==model.read_bytes(),'refuse to replace a different model'
    else:
        with model_path.open('xb') as f:
            f.write(model.read_bytes())
    mapping.append(dict(original=original['model']['path'],relocated=str(model_path.resolve()),sha256=original['model']['sha256']))
    receipt['model']=digest(model_path)
    # Provenance remains historical; relocation never implies a fresh freeze or validation.
    receipt_path=target/'freeze_receipt.json'
    encoded=(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()
    if receipt_path.exists():
        assert receipt_path.read_bytes()==encoded,'existing receipt differs; preserve it and use a fresh clone'
    else:
        with receipt_path.open('xb') as f:
            f.write(encoded)
    relocation=target/'release_relocation.json'
    if not relocation.exists():
        with relocation.open('x') as f:
            json.dump(dict(original_receipt=digest(package/'freeze_receipt.json'),mapping=mapping,
                           model_refitted=False,source_bytes_changed=False,validation_reexecuted=False),f,indent=2)
            f.write('\n')
    return receipt_path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent/'frozen/r11_kernel_match_v3')
    a=p.parse_args()
    print(install(a.root.resolve(),a.package.resolve()))
