import React from 'react'
import { NextPage } from 'next'
import Head from 'next/head'

import Playground from '@/components/playground/Playground'
import { CopilotKit } from '@copilotkit/react-core'

const Home: NextPage = () => (
  <>
    <Head>
      <link rel='icon' href='/logo.jpeg' />
      <title>Data Visualization Tool</title>
    </Head>
    <CopilotKit runtimeUrl='/api/copilotkit' agent='my_agent' showDevConsole={false}>
      <Playground />
    </CopilotKit>
  </>
)

export default Home
