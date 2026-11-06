import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { generateWallet } from '@tangle-network/tcloud/shielded'
import { createVoiceClient } from '../dist/index.js'

async function peer(context) {
  const { createServer } = await import('node:http')
  const { buildSpendAuthMessage } = await import('@tangle-network/tcloud/shielded')
  const { recoverTypedDataAddress, getAddress } = await import('viem')
  let index = 0
  createServer(async (request, response) => {
    let bytes = '', payload, verified = false
    for await (const chunk of request) bytes += chunk
    try {
      const body = JSON.parse(bytes)
      const { signature, ...fields } = body.spend_auth
      payload = fields
      const recovered = await recoverTypedDataAddress({
        ...buildSpendAuthMessage({ ...fields, serviceId: BigInt(fields.serviceId),
          amount: BigInt(fields.amount), nonce: BigInt(fields.nonce), expiry: BigInt(fields.expiry) },
          context.chainId, context.credits), signature,
      })
      const expected = context.cases[index]
      verified = getAddress(recovered) === getAddress(context.account)
        && fields.serviceId === expected.serviceId && fields.nonce === expected.nonce
        && fields.jobIndex === 0 && fields.amount === '17'
        && request.method === 'POST' && request.url === '/v1/audio/speech'
        && body.input === 'Hello' && body.voice === 'fixture'
        && body.response_format === 'mp3' && body.speed === 1
    } catch {}
    console.log(JSON.stringify({ index: index++, payload, verified }))
    response.writeHead(verified ? 200 : 422, { 'Content-Type': verified ? 'audio/mpeg' : 'application/json' })
    response.end(verified ? 'audio' : JSON.stringify({ error: { message: 'Authorization did not verify' } }))
  }).listen(0, '127.0.0.1', function () {
    console.log(JSON.stringify({ port: this.address().port }))
  })
}

const wallet = generateWallet()
const context = { account: wallet.address, chainId: 31337,
  credits: '0x0000000000000000000000000000000000000001',
  cases: [{ serviceId: '7', nonce: '3' },
    { serviceId: '9007199254740993', nonce: '9007199254740993' }] }
const child = spawn(process.execPath, ['--input-type=module', '-'], {
  cwd: new URL('..', import.meta.url), env: {}, stdio: ['pipe', 'pipe', 'inherit'],
})
const lines = createInterface({ input: child.stdout })
const captured = []
let finishCapture
const complete = new Promise(resolve => { finishCapture = resolve })
let timer
const ready = new Promise((resolve, reject) => {
  timer = setTimeout(() => reject(new Error('HTTP peer startup timed out')), 5000)
  child.once('error', reject)
  child.once('exit', () => reject(new Error('HTTP peer exited before startup')))
  lines.on('line', line => {
    const value = JSON.parse(line)
    if (value.port) { clearTimeout(timer); resolve(value.port) }
    else { captured.push(value); if (captured.length === 2) finishCapture() }
  })
})
child.stdin.end(`(${peer.toString()})(${JSON.stringify(context)}).catch(() => process.exit(1))`)
try {
  const port = await ready
  for (const example of context.cases) {
    const client = createVoiceClient({ operatorUrl: `http://127.0.0.1:${port}`,
      shieldedCreditsAddress: context.credits, chainId: context.chainId,
      commitment: wallet.commitment, spendingKeyPrivate: wallet.privateKey,
      operatorAddress: '0x0000000000000000000000000000000000000002',
      serviceId: BigInt(example.serviceId) })
    client.setNonce(BigInt(example.nonce))
    const audio = await Promise.race([
      client.synthesize('Hello', { voice: 'fixture', authorizedAmount: 17n }),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Speech request timed out')), 5000) }),
    ])
    clearTimeout(timer)
    if (Buffer.from(audio).toString() !== 'audio') throw new Error('Unexpected audio response')
  }
  await Promise.race([complete, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Authorization capture timed out')), 5000)
  })])
  if (captured.some(row => !row.verified)) throw new Error('Invalid authorization capture')
} finally {
  clearTimeout(timer)
  const exited = new Promise(resolve => child.once('exit', resolve))
  child.kill('SIGTERM')
  await Promise.race([exited, new Promise(resolve => setTimeout(resolve, 2000))])
  if (child.exitCode === null && child.signalCode === null) { child.kill('SIGKILL'); await exited }
  lines.close()
  console.log('VOICE_SDK_HTTP_RECEIPT=' + JSON.stringify({ cases: captured,
    chain_exercised: false, cleanup: child.exitCode !== null || child.signalCode !== null }))
}
