'use client'

import React from 'react'
import { useCopilotAction } from '@copilotkit/react-core'
import { CopilotChat, CopilotPopup, CopilotSidebar } from '@copilotkit/react-ui'

interface HumanChoiceProps {
  selectedQuestion: string
  setSelectedQuestion: (question: string) => void
}

export function HumanChoice({ selectedQuestion, setSelectedQuestion }: HumanChoiceProps) {
  useCopilotAction({
    name: 'human_choice',
    parameters: [
      {
        name: "feedback",
        type: "string",
        description: "The human feedback",
        required: true,
      },
    ],
    renderAndWait: ({ args, status, handler }) => (
      <EmailConfirmation
        emailContent={'Hello'}
        isExecuting={status === 'executing'}
        onRegenerate={() => handler?.('Regenerate')} // the handler is undefined while status is "executing"
        onContinue={() => handler?.('Continue')} // the handler is undefined while status is "executing"
      />
    ),
  })

  return (
    <CopilotChat
      className='h-full'
      labels={{
        title: 'Sidebar Assistant',
        initial: 'How can I help you today?',
      }}
    >
    </CopilotChat>
  )
}

interface EmailConfirmationProps {
  emailContent: string
  isExecuting: boolean
  onRegenerate: () => void
  onContinue: () => void
}

const EmailConfirmation: React.FC<EmailConfirmationProps> = ({
  emailContent,
  isExecuting,
  onRegenerate,
  onContinue,
}) => {
  return (
    <div className='p-4 bg-gray-100 rounded-lg'>
      <div className='font-bold text-lg mb-2'>Send this email?</div>
      <div className='text-gray-700'>{emailContent}</div>
      {isExecuting && (
        <div className='mt-4 flex justify-end space-x-2'>
          <button onClick={onRegenerate} className='px-4 py-2 bg-slate-400 text-white rounded'>
            Cancel
          </button>
          <button onClick={onContinue} className='px-4 py-2 bg-blue-500 text-white rounded'>
            Send
          </button>
        </div>
      )}
    </div>
  )
}
