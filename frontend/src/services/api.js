import axios from "axios";

const API_BASE = "http://localhost:8000"; // Your FastAPI backend

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return axios.post(`${API_BASE}/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const askQuestion = async (question) => {
  return axios.post(`${API_BASE}/ask`, { question: question });
};
