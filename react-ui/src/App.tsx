import { useEffect, useState } from 'react'
import { Console } from './components/console/Console'

function App() {
  const [threadId, setThreadId] = useState<string | null>(null)

  useEffect(() => {
    // Create a default thread
    const id = crypto.randomUUID()
    setThreadId(id)
  }, [])

  if (!threadId) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-900">
        <div className="text-gray-400">Loading...</div>
      </div>
    )
  }

  return <Console threadId={threadId} />
}

export default App
