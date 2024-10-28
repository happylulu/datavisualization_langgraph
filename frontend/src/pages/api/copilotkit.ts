import { NextApiRequest, NextApiResponse } from 'next'
import { CopilotRuntime, OpenAIAdapter, copilotRuntimeNextJSPagesRouterEndpoint } from '@copilotkit/runtime'
import OpenAI from 'openai'

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
const serviceAdapter = new OpenAIAdapter({ openai })

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const runtime = new CopilotRuntime({
    remoteActions: [
      {
        url: process.env.REMOTE_ACTION_URL || 'http://localhost:8000/copilotkit',
      },
    ],
  })

  const handleRequest = copilotRuntimeNextJSPagesRouterEndpoint({
    endpoint: '/api/copilotkit',
    runtime,
    serviceAdapter,
  })

  return await handleRequest(req, res)
}

export default handler
