import React, { useState } from "react";
import { uploadFile, askQuestion } from "../services/api";
import { AiOutlineLoading3Quarters } from "react-icons/ai";

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadFile(file);
      setMessages((prev) => [
        ...prev,
        { sender: "system", text: `✅ File "${file.name}" uploaded successfully.` },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "system", text: "❌ File upload failed." },
      ]);
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const question = input;
    setMessages((prev) => [...prev, { sender: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await askQuestion(question);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: res.data.answer },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "❌ Failed to get answer." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="p-4 bg-gray-800 text-lg font-bold flex items-center justify-between">
        AI Document Chatbot
        <label className="cursor-pointer bg-blue-600 px-3 py-1 rounded hover:bg-blue-700 text-sm">
          {uploading ? "Uploading..." : "📂 Upload File"}
          <input
            type="file"
            className="hidden"
            onChange={handleFileUpload}
          />
        </label>
      </div>

      {/* Messages */}
      <div className="flex-1 p-4 overflow-y-auto space-y-3">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`max-w-lg px-4 py-2 rounded-lg ${
              msg.sender === "user"
                ? "bg-blue-600 ml-auto"
                : msg.sender === "bot"
                ? "bg-gray-700"
                : "bg-gray-500 text-sm italic"
            }`}
          >
            {msg.text}
          </div>
        ))}
        {loading && (
          <div className="flex items-center space-x-2 text-gray-400">
            <AiOutlineLoading3Quarters className="animate-spin" />
            <span>Thinking...</span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 bg-gray-800 flex space-x-2">
        <input
          type="text"
          className="flex-1 px-3 py-2 rounded bg-gray-700 border border-gray-600 focus:outline-none"
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button
          onClick={handleSend}
          className="bg-green-600 px-4 py-2 rounded hover:bg-green-700"
        >
          Send
        </button>
      </div>
    </div>
  );
}
