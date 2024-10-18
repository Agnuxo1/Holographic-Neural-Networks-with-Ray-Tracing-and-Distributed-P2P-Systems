"use client"

import React, { useState, useEffect, useRef, useMemo } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Html, useTexture, Stars, Trail } from '@react-three/drei'
import { EffectComposer, SSR, Bloom, DepthOfField, Noise } from '@react-three/postprocessing'
import { KernelSize } from 'postprocessing'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, Download, Upload } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { HfInference } from '@huggingface/inference'
import axios from 'axios'
import * as pdfjs from 'pdfjs-dist'

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

// NVIDIA API placeholders
const LLAMA_INDEX_API = "YOUR-LLAMA-INDEX-API-KEY-HERE"
const NEMOTRON_70B_API = "YOUR-NEMOTRON-70B-API-KEY-HERE"
const RAG_NVIDIA_API = "YOUR-RAG-NVIDIA-API-KEY-HERE"
const NEMO_GUARDRAILS_API = "YOUR-NEMO-GUARDRAILS-API-KEY-HERE"

// Global variable to control NVIDIA API usage
const useNvidiaAPIs = false

// Enhanced holographic shader
const holographicVertexShader = `
varying vec2 vUv;
varying vec3 vPosition;
void main() {
  vUv = uv;
  vPosition = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const holographicFragmentShader = `
uniform float time;
uniform vec3 color;
varying vec2 vUv;
varying vec3 vPosition;

float noise(vec3 p) {
  return fract(sin(dot(p, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
}

void main() {
  vec3 pos = vPosition * 2.0;
  float n = noise(pos + time * 0.1);
  
  float intensity = 1.0 - length(vUv - 0.5) * 2.0;
  intensity = pow(intensity, 3.0);
  
  vec3 glow = color * intensity * (0.5 + 0.5 * sin(time + vUv.x * 10.0));
  
  gl_FragColor = vec4(glow + vec3(n * 0.1), intensity * 0.8);
}
`

function Neuron({ position, activation, relevance, isContextNode }) {
const meshRef = useRef()
const [hovered, setHovered] = useState(false)
const texture = useTexture('/placeholder.svg?height=128&width=128')

const color = isContextNode 
  ? new THREE.Color(relevance, 0, 1 - relevance)
  : new THREE.Color(activation, 0, 1 - activation)

useFrame((state) => {
  if (meshRef.current) {
    meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.2
    meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.2
  }
})

return (
  <group position={position}>
    <mesh
      ref={meshRef}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[isContextNode ? 0.2 : 0.15, 32, 32]} />
      <shaderMaterial
        vertexShader={holographicVertexShader}
        fragmentShader={holographicFragmentShader}
        uniforms={{
          time: { value: 0 },
          color: { value: color }
        }}
        transparent
      />
    </mesh>
    <Trail
      width={0.05}
      length={5}
      color={color}
      attenuation={(t) => t * t}
    >
      <mesh>
        <sphereGeometry args={[0.02, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </Trail>
    {hovered && (
      <Html distanceFactor={10}>
        <div className="bg-black bg-opacity-75 text-white p-2 rounded text-xs">
          {isContextNode ? `Relevance: ${relevance.toFixed(2)}` : `Activation: ${activation.toFixed(2)}`}
        </div>
      </Html>
    )}
  </group>
)
}

function Connection({ start, end, strength }) {
const points = useMemo(() => [
  new THREE.Vector3(...start),
  new THREE.Vector3(...end)
], [start, end])

return (
  <mesh>
    <tubeGeometry args={[new THREE.CatmullRomCurve3(points), 64, 0.01, 8, false]} />
    <meshBasicMaterial color={new THREE.Color(strength, 0, 1 - strength)} transparent opacity={0.3} />
  </mesh>
)
}

function NeuralNetwork({ neurons, connections, contextNodes }) {
return (
  <group>
    {neurons.map((neuron, i) => (
      <Neuron key={i} position={neuron.position} activation={neuron.activation} isContextNode={false} />
    ))}
    {contextNodes.map((node, i) => (
      <Neuron key={`context-${i}`} position={node.position} relevance={node.relevance} isContextNode={true} />
    ))}
    {connections.map((connection, i) => (
      <Connection
        key={i}
        start={neurons[connection.start].position}
        end={neurons[connection.end].position}
        strength={connection.strength}
      />
    ))}
  </group>
)
}

function HolographicPlane() {
const shaderRef = useRef()
const texture = useTexture('/placeholder.svg?height=1024&width=1024')

useFrame(({ clock }) => {
  if (shaderRef.current) {
    shaderRef.current.uniforms.time.value = clock.getElapsedTime()
  }
})

return (
  <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -10, 0]}>
    <planeGeometry args={[100, 100]} />
    <shaderMaterial
      ref={shaderRef}
      vertexShader={holographicVertexShader}
      fragmentShader={holographicFragmentShader}
      uniforms={{
        time: { value: 0 },
        color: { value: new THREE.Color(0.1, 0.3, 0.6) }
      }}
      transparent
    />
  </mesh>
)
}

function Scene({ neurons, connections, contextNodes }) {
const { camera } = useThree()

useEffect(() => {
  camera.position.set(0, 20, 40)
}, [camera])

return (
  <>
    <ambientLight intensity={0.1} />
    <pointLight position={[10, 10, 10]} intensity={0.5} />
    <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
    <NeuralNetwork neurons={neurons} connections={connections} contextNodes={contextNodes} />
    <HolographicPlane />
    <EffectComposer>
      <SSR intensity={0.45} exponent={1} distance={10} fade={10} roughnessFade={1} thickness={10} ior={0.45} maxRoughness={1} maxDepthDifference={10} blend={0.95} correction={1} correctionRadius={1} blur={0} blurKernel={1} blurSharpness={10} jitter={0.75} jitterRoughness={0.2} steps={40} refineSteps={5} missedRays={true} useNormalMap={true} useRoughnessMap={true} resolutionScale={1} velocityResolutionScale={1} />
      <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} />
      <Noise opacity={0.02} />
      <DepthOfField focusDistance={0} focalLength={0.02} bokehScale={2} height={480} />
    </EffectComposer>
    <OrbitControls />
  </>
)
}

class HolographicNeuralNetwork {
constructor(numNeurons) {
  this.neurons = Array.from({ length: numNeurons }, () => ({
    position: [
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20
    ],
    activation: 0
  }))
  this.connections = this.initializeConnections()
  this.knowledgeBase = {}
  this.contextNodes = []
}

initializeConnections() {
  const connections = []
  for (let i = 0; i < this.neurons.length; i++) {
    for (let j = i + 1; j < this.neurons.length; j++) {
      if (Math.random() < 0.1) {  // 10% chance of connection
        connections.push({ start: i, end: j, strength: Math.random() })
      }
    }
  }
  return connections
}

activate(input) {
  const inputHash = this.hash(input)
  const initialNeuron = inputHash % this.neurons.length

  this.neurons.forEach((neuron, i) => {
    const distance = Math.sqrt(
      this.neurons[initialNeuron].position.reduce((sum, coord, j) => 
        sum + Math.pow(coord - neuron.position[j], 2), 0)
    )
    neuron.activation = Math.exp(-distance / 10)  // Activation decays with distance
  })

  // Propagate activations through connections
  this.connections.forEach(conn => {
    const sourceActivation = this.neurons[conn.start].activation
    this.neurons[conn.end].activation += sourceActivation * conn.strength
  })

  // Normalize activations
  const maxActivation = Math.max(...this.neurons.map(n => n.activation))
  this.neurons.forEach(n => n.activation /= maxActivation)

  return this.neurons.map(n => n.activation)
}

learn(input, output) {
  const activations = this.activate(input)
  this.knowledgeBase[input] = { output, activations }
}

generateResponse(input) {
  const activations = this.activate(input)
  const similarities = Object.entries(this.knowledgeBase).map(([key, value]) => ({
    key,
    similarity: this.cosineSimilarity(activations, value.activations)
  }))
  similarities.sort((a, b) => b.similarity - a.similarity)
  return similarities[0] ? this.knowledgeBase[similarities[0].key].output : "I don't know how to respond to that."
}

updateContextNodes(ragContext) {
  this.contextNodes = ragContext.map(node => ({
    position: [
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20
    ],
    relevance: node.score
  }))
}

hash(input) {
  return input.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
}

cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0)
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
  return dotProduct / (magnitudeA * magnitudeB)
}

exportKnowledge() {
  return JSON.stringify(this.knowledgeBase)
}

importKnowledge(knowledge) {
  try {
    this.knowledgeBase = JSON.parse(knowledge)
    return true
  } catch (error) {
    console.error("Error importing knowledge:", error)
    return false
  }
}

// Methods for NVIDIA API integration
async useLlamaIndex(input) {
  if (LLAMA_INDEX_API !== "YOUR-LLAMA-INDEX-API-KEY-HERE") {
    const response = await axios.post(LLAMA_INDEX_API, { query: input })
    return response.data.result
  }
  return null
}

async useNemotron70B(input) {
  if (NEMOTRON_70B_API !== "YOUR-NEMOTRON-70B-API-KEY-HERE") {
    const response = await axios.post(NEMOTRON_70B_API, { text: input })
    return response.data.generated_text
  }
  return null
}

async useRagNvidia(input) {
  if (RAG_NVIDIA_API !== "YOUR-RAG-NVIDIA-API-KEY-HERE") {
    const response = await axios.post(RAG_NVIDIA_API, { query: input })
    return response.data.result
  }
  return null
}

async useNemoGuardrails(input) {
  if (NEMO_GUARDRAILS_API !== "YOUR-NEMO-GUARDRAILS-API-KEY-HERE") {
    const response = await axios.post(NEMO_GUARDRAILS_API, { text: input })
    return response.data.safe_text
  }
  return null
}

// Method for generating multiple words
generateWords(input, count = 5) {
  let currentWord = input
  const words = [currentWord]

  for (let i = 1; i < count; i++) {
    const nextWord = this.generateResponse(currentWord)
    if (nextWord === "I don't know how to respond to that.") {
      break
    }
    words.push(nextWord)
    currentWord = nextWord
  }

  return words
}
}

export default function UnifiedHolographicNeuralNetwork() {
const [chatHistory, setChatHistory] = useState([])
const [inputText, setInputText] = useState('')
const [hnn, setHnn] = useState(null)
const  [isLLMActive, setIsLLMActive] = useState(false)
const [llmModel, setLLMModel] = useState(null)
const [error, setError] = useState(null)
const [isTraining, setIsTraining] = useState(false)
const [trainingProgress, setTrainingProgress] = useState(0)
const [learnInput, setLearnInput] = useState('')
const [learnResponse, setLearnResponse] = useState('')
const [isProcessing, setIsProcessing] = useState(false)
const [progress, setProgress] = useState(0)
const [predictedWords, setPredictedWords] = useState([])
const [nodeId, setNodeId] = useState('')
const [p2pIp, setP2pIp] = useState('')
const [p2pPort, setP2pPort] = useState('')
const [p2pConnected, setP2pConnected] = useState(false)

useEffect(() => {
  const newHnn = new HolographicNeuralNetwork(100)  // 100 neurons
  setHnn(newHnn)

  // Initialize LLM model
  const hf = new HfInference("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")
  setLLMModel(hf)
  setIsLLMActive(true)

  // Attempt to create account automatically
  createAccountAutomatically()
}, [])

const handleSubmit = async (e) => {
  e.preventDefault()
  if (!inputText.trim() || !hnn) return

  let response
  let ragContext = []
  if (isLLMActive && llmModel) {
    try {
      if (useNvidiaAPIs) {
        // Use NVIDIA APIs
        const llamaIndexResult = await hnn.useLlamaIndex(inputText)
        const nemotronResult = await hnn.useNemotron70B(inputText)
        const ragResult = await hnn.useRagNvidia(inputText)
        const guardrailsResult = await hnn.useNemoGuardrails(inputText)

        // Combine results (this is a simple example, you might want to implement a more sophisticated combination strategy)
        response = `LlamaIndex: ${llamaIndexResult}
Nemotron: ${nemotronResult}
RAG: ${ragResult}
Guardrails: ${guardrailsResult}`
      } else {
        // Use existing functionality
        ragContext = [
          { text: "Relevant context 1", score: 0.9 },
          { text: "Relevant context 2", score: 0.7 },
          { text: "Relevant context 3", score: 0.5 }
        ]
        hnn.updateContextNodes(ragContext)

        const llmResponse = await llmModel.textGeneration({
          model: 'facebook/opt-350m',
          inputs: inputText,
          parameters: {
            max_new_tokens: 50,
            temperature: 0.7,
            top_p: 0.95,
            repetition_penalty: 1.1,
          },
        })
        response = llmResponse.generated_text
      }
    } catch (error) {
      console.error('Error generating response:', error)
      response = "Sorry, I couldn't generate a response."
    }
  } else {
    response = hnn.generateResponse(inputText)
  }

  hnn.learn(inputText, response)
  setChatHistory(prev => [...prev, { type: 'user', text: inputText }, { type: 'bot', text: response }])
  
  setInputText('')

  // Store data in P2P network
  if (p2pConnected) {
    try {
      await axios.post('http://localhost:3001/api/store', { key: inputText, value: response })
    } catch (error) {
      console.error('Error storing data in P2P network:', error)
    }
  }

  // Update predicted words
  const words = hnn.generateWords(inputText.split(' ').pop(), 10)
  setPredictedWords(words)
}

const createAccountAutomatically = async () => {
  try {
    // Request permission to access user's Google account
    const permission = await requestGoogleAccountPermission()
    if (permission) {
      // Generate a unique node ID based on the user's Google account
      const googleAccountInfo = await getGoogleAccountInfo()
      const generatedNodeId = generateNodeIdFromGoogleAccount(googleAccountInfo)
      
      // Create account using the generated node ID
      const response = await axios.post('http://localhost:3001/api/create-account', { nodeId: generatedNodeId })
      console.log('Account created automatically:', response.data)
      setNodeId(response.data.nodeId)
      
      // Attempt to connect to P2P network
      await handleConnectP2P()
    } else {
      console.log('Permission to access Google account denied')
    }
  } catch (error) {
    console.error('Error creating account automatically:', error)
    setError('Failed to create account automatically')
  }
}

const requestGoogleAccountPermission = async () => {
  // This is a placeholder function. In a real implementation, you would use the Google Identity Services API
  // to request permission to access the user's Google account.
  return new Promise((resolve) => {
    const granted = window.confirm("Do you want to grant permission to access your Google account for automatic account creation?")
    resolve(granted)
  })
}

const getGoogleAccountInfo = async () => {
  // This is a placeholder function. In a real implementation, you would use the Google Identity Services API
  // to retrieve the user's Google account information.
  return {
    id: "placeholder_google_id",
    email: "placeholder@gmail.com"
  }
}

const generateNodeIdFromGoogleAccount = (accountInfo) => {
  // This is a simple example. In a real implementation, you might want to use a more sophisticated method
  // to generate a unique node ID based on the user's Google account information.
  return `node_${accountInfo.id}_${Date.now()}`
}

const handleConnectP2P = async () => {
  try {
    // In a real implementation, you might want to use a service discovery mechanism to find available P2P nodes
    const defaultP2PIP = "localhost"
    const defaultP2PPort = 3001

    await axios.post('http://localhost:3001/api/connect', { ip: defaultP2PIP, port: defaultP2PPort })
    setP2pConnected(true)
    console.log('Connected to P2P network')
  } catch (error) {
    console.error('Error connecting to P2P network:', error)
    setError('Failed to connect to P2P network')
  }
}

const handleLearn = () => {
  if (learnInput.trim() && learnResponse.trim() && hnn) {
    hnn.learn(learnInput, learnResponse)
    setLearnInput('')
    setLearnResponse('')
    alert('Learning completed')
  }
}

const handleSave = () => {
  if (hnn) {
    const knowledge = hnn.exportKnowledge()
    const blob = new Blob([knowledge], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'holographic_nn_knowledge.json'
    a.click()
  }
}

const handleLoad = (event) => {
  const file = event.target.files?.[0]
  if (file && hnn) {
    const reader = new FileReader()
    reader.onload = (e) => {
      const knowledge = e.target?.result
      if (typeof knowledge === 'string') {
        const success = hnn.importKnowledge(knowledge)
        if (success) {
          alert('Knowledge loaded successfully')
        } else {
          alert('Error loading knowledge')
        }
      }
    }
    reader.readAsText(file)
  }
}

const handleTrain = () => {
  setIsTraining(true)
  setTrainingProgress(0)
  const trainStep = (i) => {
    if (i >= 100) {  // Train on 100 random samples
      setIsTraining(false)
      setTrainingProgress(0)
      alert('Training completed successfully')
      return
    }

    const input = `Random input ${i}`
    const output = `Random output ${i}`
    hnn.learn(input, output)
    setTrainingProgress(Math.round(((i + 1) / 100) * 100))

    setTimeout(() => trainStep(i + 1), 100)
  }

  trainStep(0)
}

const processPDF = async (file) => {
  const arrayBuffer = await file.arrayBuffer()
  const pdf = await pdfjs.getDocument(arrayBuffer).promise
  let text = ''
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()
    text += content.items.map((item) => item.str).join(' ') + ' '
    setProgress((i / pdf.numPages) * 100)
  }
  return text
}

const handleFileUpload = async (event) => {
  const file = event.target.files?.[0]
  if (file) {
    setIsProcessing(true)
    setProgress(0)
    try {
      let text = ''
      if (file.name.endsWith('.txt')) {
        text = await file.text()
      } else if (file.name.endsWith('.pdf')) {
        text = await processPDF(file)
      } else {
        throw new Error('Unsupported file type')
      }

      const words = text.toLowerCase().match(/\b\w+\b/g) || []
      for (let i = 0; i < words.length - 1; i++) {
        hnn.learn(words[i], words[i + 1])
      }

      setIsProcessing(false)
      setError(null)
      alert('File processed successfully')
    } catch (err) {
      setError('Error processing file. Please try again.')
      setIsProcessing(false)
    }
  }
}

return (
  <div className="flex h-screen">
    <div className="w-1/2 p-4 overflow-y-auto">
      <Card>
        <CardHeader>
          <CardTitle>Holographic Neural Network Chat</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 mb-4">
            {chatHistory.map((message, index) => (
              <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-xs p-2 rounded-lg ${message.type === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
                  {message.text}
                </div>
              </div>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <Input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type your message..."
              className="flex-grow"
            />
            <Button type="submit">Send</Button>
          </form>
          {predictedWords.length > 0 && (
            <div className="mt-2 text-sm text-gray-500">
              Predicted words: {predictedWords.join(', ')}
            </div>
          )}
        </CardContent>
      </Card>
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>P2P Network</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Input
              type="text"
              value={nodeId}
              onChange={(e) => setNodeId(e.target.value)}
              placeholder="Enter Node ID"
              disabled
            />
            <Button onClick={createAccountAutomatically}>Create Account Automatically</Button>
            <Input
              type="text"
              value={p2pIp}
              onChange={(e) => setP2pIp(e.target.value)}
              placeholder="Enter P2P IP"
            />
            <Input
              type="text"
              value={p2pPort}
              onChange={(e) => setP2pPort(e.target.value)}
              placeholder="Enter P2P Port"
            />
            <Button onClick={handleConnectP2P} disabled={p2pConnected}>
              {p2pConnected ? 'Connected' : 'Connect to P2P Network'}
            </Button>
          </div>
        </CardContent>
      </Card>
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Learning</CardTitle>
        </CardHeader>
        <CardContent>
          <Input
            type="text"
            value={learnInput}
            onChange={(e) => setLearnInput(e.target.value)}
            placeholder="Input to learn"
            className="mb-2"
          />
          <Input
            type="text"
            value={learnResponse}
            onChange={(e) => setLearnResponse(e.target.value)}
            placeholder="Associated response"
            className="mb-2"
          />
          <Button onClick={handleLearn}>Learn</Button>
        </CardContent>
      </Card>
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Knowledge Management</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2">
            <Button onClick={handleSave}>
              <Download className="mr-2 h-4 w-4" />
              Save Knowledge
            </Button>
            <Input type="file" accept=".json" onChange={handleLoad} className="hidden" id="load-knowledge" />
            <Button onClick={() => document.getElementById('load-knowledge')?.click()}>
              <Upload className="mr-2 h-4 w-4" />
              Load Knowledge
            </Button>
          </div>
        </CardContent>
      </Card>
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Training</CardTitle>
        </CardHeader>
        <CardContent>
          <Button onClick={handleTrain} disabled={isTraining}>
            {isTraining ? 'Training...' : 'Train'}
          </Button>
          {isTraining && (
            <div className="mt-2">
              <Progress value={trainingProgress} className="w-full" />
              <p className="text-center mt-2">{trainingProgress}% completed</p>
            </div>
          )}
        </CardContent>
      </Card>
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>File Processing</CardTitle>
        </CardHeader>
        <CardContent>
          <Input type="file" onChange={handleFileUpload} accept=".txt,.pdf" />
          {isProcessing && (
            <div className="mt-2">
              <Progress value={progress} className="w-full" />
              <p className="text-center mt-2">{Math.round(progress)}% processed</p>
            </div>
          )}
        </CardContent>
      </Card>
      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
    <div className="w-1/2">
      <Canvas>
        <Scene 
          neurons={hnn ? hnn.neurons : []} 
          connections={hnn ? hnn.connections : []} 
          contextNodes={hnn ? hnn.contextNodes : []}
        />
      </Canvas>
    </div>
  </div>
)
}