/* eslint-disable react/jsx-props-no-spreading */
import React from 'react'
import { AppProps } from 'next/app'

import '@/styles/tailwind.css'
import '@copilotkit/react-ui/styles.css'
import '@/styles/copilotkit.css'

const App = ({ Component, pageProps }: AppProps) => <Component {...pageProps} />

export default App
