import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

function json(name){return JSON.parse(readFileSync(new URL('../'+name, import.meta.url), 'utf8'));}
test('P15: ferramentas de CI são isoladas do pacote JS e possuem lockfile de integridade',()=>{
  const core=json('package.json');
  const pkg=json('ci-tools/package.json');
  const lock=json('ci-tools/package-lock.json');
  assert.equal(core.dependencies,undefined);
  assert.equal(core.devDependencies,undefined);
  assert.equal(core.private,true);
  assert.equal(pkg.private,true);
  assert.equal(lock.lockfileVersion,3);
  assert.deepEqual(lock.packages[''].devDependencies,pkg.devDependencies);
  assert.equal(pkg.devDependencies.playwright,'1.64.1');
  assert.equal(pkg.devDependencies.wrangler,'4.152.0');
  assert.ok(Object.keys(lock.packages).length>=80);
  for(const [name,metadata] of Object.entries(lock.packages)){
    if(!name) continue;
    if(metadata.resolved && metadata.resolved.startsWith('https://'))
      assert.match(metadata.integrity||'', /^sha512-/,name);
  }
  const workflow=readFileSync(new URL('../.github/workflows/ci.yml',import.meta.url),'utf8');
  assert.match(workflow,/npm ci --prefix ci-tools --no-audit --no-fund/);
  assert.doesNotMatch(workflow,/npm install --no-save/);
  assert.doesNotMatch(workflow,/ci-lock-generator:/);
  assert.match(workflow,/PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD/);
});
