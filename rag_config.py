"""
Configuración del sistema RAG para TUPA - Versión Streamlit Cloud
Este archivo maneja la configuración tanto para desarrollo local como Streamlit Cloud
"""

import os
import streamlit as st
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuración centralizada para el sistema RAG"""
    
    def __init__(self):
        # Intentar obtener configuración de Streamlit secrets primero, luego variables de entorno
        try:
            # Streamlit Cloud secrets
            self.openai_api_key = st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
            self.pinecone_api_key = st.secrets.get("pinecone_api_key", os.getenv("PINECONE_API_KEY", ""))
            self.pinecone_environment = st.secrets.get("pinecone_environment", os.getenv("PINECONE_ENVIRONMENT", ""))
            self.pinecone_index_name = st.secrets.get("pinecone_index_name", os.getenv("PINECONE_INDEX_NAME", "tupa-cusco"))
            
            # Configuración del modelo
            self.embedding_model = st.secrets.get("embedding_model", os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
            self.chat_model = st.secrets.get("chat_model", os.getenv("CHAT_MODEL", "gpt-3.5-turbo"))
            self.max_tokens = int(st.secrets.get("max_tokens", os.getenv("MAX_TOKENS", "1000")))
            self.temperature = float(st.secrets.get("temperature", os.getenv("TEMPERATURE", "0.7")))
            
            # Configuración de la app
            self.app_title = st.secrets.get("app_title", os.getenv("APP_TITLE", "Asistente TUPA RAG"))
            self.app_description = st.secrets.get("app_description", os.getenv("APP_DESCRIPTION", "Sistema RAG con Pinecone para consultas TUPA"))
            
        except Exception:
            # Fallback a variables de entorno si no están disponibles los secrets
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
            self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
            self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "tupa-cusco")
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.chat_model = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
            self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
            self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
            self.app_title = os.getenv("APP_TITLE", "Asistente TUPA RAG")
            self.app_description = os.getenv("APP_DESCRIPTION", "Sistema RAG con Pinecone para consultas TUPA")
        
        # Parámetros fijos del sistema
        self.embedding_dimension = 384  # Para all-MiniLM-L6-v2
        self.top_k_results = 5
        self.similarity_threshold = 0.7
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # Validar configuración
        self._validate_config()
    
    def _validate_config(self):
        """Validación de configuración con mensajes útiles"""
        missing_configs = []
        
        if not self.openai_api_key:
            missing_configs.append("OpenAI API Key")
        if not self.pinecone_api_key:
            missing_configs.append("Pinecone API Key")
        if not self.pinecone_environment:
            missing_configs.append("Pinecone Environment")
        
        if missing_configs:
            error_msg = f"Configuración faltante: {', '.join(missing_configs)}"
            st.error(f"❌ {error_msg}")
            st.error("🔧 Configura estos valores en Streamlit Cloud > Settings > Secrets")
            raise ValueError(error_msg)
    
    def get_status(self):
        """Obtiene estado de la configuración"""
        return {
            "openai_configured": bool(self.openai_api_key),
            "pinecone_configured": bool(self.pinecone_api_key and self.pinecone_environment),
            "model": self.chat_model,
            "embedding_model": self.embedding_model,
            "index_name": self.pinecone_index_name
        }

# Instancia global de configuración
try:
    rag_config = RAGConfig()
except Exception as e:
    st.error(f"❌ Error de configuración: {e}")
    st.stop()
