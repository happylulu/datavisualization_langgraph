import React, { useState, useEffect, useCallback, useRef } from 'react'
import Form from './Form'
import Logo from '../Logo'
import { Client } from '@langchain/langgraph-sdk'
import { QuestionDisplay } from './QuestionDisplay'
import { Stream } from './Stream'
import { graphDictionary, InputType } from '../graphs/graphDictionary'
import UploadButton from '../UploadButton'
import { Sidebar } from './Sidebar'
import { Button } from '@mui/material'

type GraphComponentProps = InputType & { data: any }

const sampleQuestions = [
  'Is the study progressing according to the planned timeline (e.g., enrollment rate, follow-up visits)?',
  'Are adverse events (AEs) and serious adverse events (SAEs) within expected safety parameters?',
  'Are the enrolled subjects representative of the intended population in terms of key demographics (age, gender, race, etc.)?',
  'What are the reasons behind patient dropouts, and are they preventable or related to the treatment?',
  'Are the primary and secondary endpoints being met at interim stages?',
  'Are there any treatment-related adverse events that were unexpected or more severe than predicted?',
  'Are there missing data points, and if so, how are they being handled?',
  'Does the interim analysis suggest continuing, stopping for futility, or stopping for early success?',
]
export type GraphState = {
  messages: [{ [key: string]: any }]
  hypothesis: { [key: string]: any }
  process_decision: string
  process: string
  visualization_state: string
  searcher_state: string
  code_state: string
  report_section: string
  quality_review: string
  needs_revision: boolean
  last_sender: string
  error: string
}

export default function Playground() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [selectedQuestion, setSelectedQuestion] = useState('')
  const [displayedQuestions, setDisplayedQuestions] = useState<string[]>([])
  const [graphState, setGraphState] = useState<GraphState | null>(null)
  const [databaseUuid, setDatabaseUuid] = useState<string | null>(null)
  const [databaseFileName, setDatabaseFileName] = useState<string | null>(null)
  const [showSidebar, setShowSidebar] = useState(false)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [showHumanFeedbackForm, setShowHumanFeedbackForm] = useState(false)
  const [threadId, setThreadId] = useState<string | null>(null)

  const uploadDatabase = useCallback(async (file: File): Promise<string> => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_SQLITE_URL + '/upload-file', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      return data.uuid
    } catch (error) {
      console.error('Error uploading file:', error)
      throw error
    }
  }, [])

  const run = useCallback(
    async (question: string) => {
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question, databaseUuid }),
      })

      if (!response.ok) {
        throw new Error('Run failed')
      }

      const reader = response.body?.getReader()
      if (!reader) return

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = new TextDecoder().decode(value)
        const lines = chunk.split('\n\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const { data, threadId } = JSON.parse(line.slice(6))

            setThreadId(threadId)
            if (data.error && data.message === "Invalid choice. Please provide '1' or '2'.")
              setShowHumanFeedbackForm(true)

            setGraphState((prev) => ({ ...prev, ...data }))
            console.log(data)
          }
        }
      }
    },
    [databaseUuid],
  )

  const handleFileUpload = useCallback(
    async (file: File) => {
      setIsUploading(true)
      try {
        const uuid = await uploadDatabase(file)
        setDatabaseUuid(uuid)
        setDatabaseFileName(file.name)
        console.log(`File "${file.name}" uploaded successfully. UUID: ${uuid}`)
      } catch (error) {
        console.error('Failed to upload file:', error)
        alert('Failed to upload file')
      } finally {
        setIsUploading(false)
      }
    },
    [uploadDatabase, setDatabaseUuid, setDatabaseFileName],
  )

  useEffect(() => {
    const rotateInterval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % sampleQuestions.length)
    }, 3000)

    return () => clearInterval(rotateInterval)
  }, [])

  useEffect(() => {
    const startIndex = currentIndex
    const endIndex = (currentIndex + 5) % sampleQuestions.length
    if (startIndex < endIndex) {
      setDisplayedQuestions(sampleQuestions.slice(startIndex, endIndex))
    } else {
      setDisplayedQuestions([...sampleQuestions.slice(startIndex), ...sampleQuestions.slice(0, endIndex)])
    }
  }, [currentIndex])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target as Node)) {
        setShowSidebar(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  const handleQuestionClick = (question: string) => {
    setSelectedQuestion(question)
  }

  const onFormSubmit = useCallback(async () => {
    await run(selectedQuestion)
  }, [run, selectedQuestion])

  const toggleSidebar = () => {
    setShowSidebar(!showSidebar)
  }

  const handleHumanFeedback = async (approval: boolean) => {
    const client = new Client({
      apiKey: process.env.LANGSMITH_API_KEY,
      apiUrl: process.env.LANGGRAPH_API_URL,
    })

    if (threadId) {
      // const state = await client.threads.getState(threadId)
      // const toolCallId = state.values.messages[state.values.messages.length - 1].tool_calls[0].id

      // // We now create the tool call with the id and the response we want
      // const toolMessage = [
      //   {
      //     tool_call_id: toolCallId,
      //     type: 'tool',
      //     content: approval ? '2' : '1',
      //   },
      // ]

      // await client.threads.updateState(threadId, { values: { messages: toolMessage }, asNode: 'human_choice_node' })
      console.log(approval, threadId)
    }
  }

  return (
    <div className='flex flex-col items-center justify-center min-h-screen bg-[#204544] m-0 p-0'>
      <Logo setGraphState={setGraphState} />
      <UploadButton onFileUpload={handleFileUpload} disabled={isUploading} />

      <Form
        selectedQuestion={selectedQuestion}
        setSelectedQuestion={setSelectedQuestion}
        onFormSubmit={onFormSubmit}
        disabled={isUploading}
      />

      {!graphState && (
        <>
          <div className='text-white text-center mb-20 w-2/3'>
            Don't have a .sqlite or .csv file to query? We'll use this one by default:{' '}
            <a
              href='https://docs.google.com/spreadsheets/d/1S2mYAKwYYmjZW6jURiAfMWTVmwg74QQDfwdMUvVEgMk/edit?usp=sharing'
              target='_blank'
              rel='noopener noreferrer'
              className='text-blue-300 hover:text-blue-100'
            >
              Sample Dataset
            </a>
          </div>
          <QuestionDisplay displayedQuestions={displayedQuestions} handleQuestionClick={handleQuestionClick} />
        </>
      )}

      {graphState && graphState.visualization_state === '' && (
        <div className='flex  w-2/3 items-start  items-center justify-center mt-60'>
          <Stream graphState={graphState} />
        </div>
      )}
      {showHumanFeedbackForm && (
        <div className='flex justify-end gap-2 mb-10'>
          <Button variant='contained' color='success' onClick={(ev) => handleHumanFeedback(true)}>
            Approve
          </Button>
          <Button variant='contained' color='error' onClick={(ev) => handleHumanFeedback(false)}>
            Reject
          </Button>
        </div>
      )}
      {/* {graphState && graphState.visualization == 'none' && (
        <div id='answer_canvas' className='p-10 w-2/3 flex flex-col items-center justify-center relative'>
          <button
            onClick={toggleSidebar}
            className='absolute top-12 right-12 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded'
          >
            See Traces
          </button>
          <div className='flex w-full flex-col p-10 rounded-[10px] bg-white items-center justify-center'>
            <div className='text-lg mx-20'>{graphState.answer}</div>
            {graphState.visualization_reason && (
              <div className='text-sm mt-10 text-gray-500 mx-20'>{graphState.visualization_reason}</div>
            )}
          </div>
          {showSidebar && (
            <div ref={sidebarRef}>
              <Sidebar graphState={graphState} onClose={toggleSidebar} />
            </div>
          )}
        </div>
      )}

      {graphState && graphState.formatted_data_for_visualization && (
        <div id='answer_canvas' className='p-10 w-full flex flex-col items-center justify-center relative'>
          <button
            onClick={toggleSidebar}
            className='absolute top-12 right-12 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded'
          >
            See Traces
          </button>
          <div className='flex w-full flex-col p-10 rounded-[10px] bg-white items-center justify-center'>
            <div className='text-sm mb-10 mx-20'>
              {graphState.answer && <div className='markdown-content'>{graphState.answer}</div>}
            </div>
            {React.createElement(
              graphDictionary[graphState.visualization as keyof typeof graphDictionary]
                .component as React.ComponentType<any>,
              {
                data: graphState.formatted_data_for_visualization,
              },
            )}
          </div>
          {showSidebar && (
            <div ref={sidebarRef}>
              <Sidebar graphState={graphState} onClose={toggleSidebar} />
            </div>
          )}
        </div>
      )} */}
    </div>
  )
}
