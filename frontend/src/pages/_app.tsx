/* eslint-disable react/jsx-props-no-spreading */
import React from 'react'
import { AppProps } from 'next/app'
import { ToastContainer } from 'react-toastify'

import '@/styles/tailwind.css'
import '@copilotkit/react-ui/styles.css'
import '@/styles/copilotkit.css'
import 'react-toastify/dist/ReactToastify.css'

const App = ({ Component, pageProps }: AppProps) => (
  <>
    <Component {...pageProps} />
    <ToastContainer position='top-right' autoClose={5000} />
  </>
)

export default App
