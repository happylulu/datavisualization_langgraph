import { StreamRow } from './StreamRow'
import { GraphState } from './Playground'

export const Stream = ({ graphState }: { graphState: GraphState }) => {

  return (
    <div className='w-full mb-10 items-center  '>
      {graphState.hypothesis && <StreamRow heading='Hypothesis' information={graphState.hypothesis.content} />}
    </div>
  )
}
