import streamlit as st
import time
import logging
from typing import Optional

# Imports del sistema RAG
try:
    from rag_config import rag_config
    from rag_system import rag_system, RAGResponse
    from pinecone_client import pinecone_client
    from document_processor import document_processor, create_sample_tupa_documents
except ImportError as e:
    st.error(f"Error importando módulos RAG: {e}")
    st.error("Asegúrate de que todos los archivos RAG estén en el directorio")
    st.stop()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
st.set_page_config(
    page_title=rag_config.app_title,
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# DISEÑO PROFESIONAL
# ---------------------------
st.markdown("""
<style>
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary-color: #64748b;
        --success-color: #059669;
        --background: #ffffff;
        --surface: #f8fafc;
        --border: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius: 12px;
    }

    .stApp {
        background: var(--background);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }
    
    .mini-header {
        text-align: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.95);
    }
    
    .mini-header h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .mini-header p {
        margin: 0.1rem 0 0 0;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    
    .hero-section {
        text-align: center;
        padding: 4rem 2rem 3rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 1rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        margin: 0 0 0.5rem 0;
        font-weight: 400;
    }
    
    .hero-description {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin: 0;
        font-weight: 300;
        opacity: 0.8;
    }

    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .stChatMessage {
        background: transparent !important;
        padding: 1.5rem 0 !important;
        border: none !important;
        margin: 0 !important;
    }
    
    .stChatMessage[data-testid="user-message"] > div {
        background: var(--primary-color) !important;
        color: white !important;
        padding: 1rem 1.5rem !important;
        border-radius: 18px 18px 4px 18px !important;
        margin-left: auto !important;
        max-width: 70% !important;
        box-shadow: var(--shadow-sm) !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] > div {
        background: var(--surface) !important;
        color: var(--text-primary) !important;
        padding: 1.5rem !important;
        border-radius: 18px 18px 18px 4px !important;
        margin-right: auto !important;
        max-width: 85% !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    .stChatInputContainer {
        background: white !important;
        border: 2px solid var(--border) !important;
        border-radius: 24px !important;
        box-shadow: var(--shadow-lg) !important;
        transition: all 0.2s ease !important;
        max-width: 800px !important;
        margin: 2rem auto !important;
    }
    
    .stChatInputContainer:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1), var(--shadow-lg) !important;
    }
    
    .stChatInputContainer input {
        background: transparent !important;
        border: none !important;
        font-size: 1rem !important;
        padding: 1rem 1.5rem !important;
        color: var(--text-primary) !important;
    }
    
    .stChatInputContainer input::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.7 !important;
    }
    
    .stChatInputContainer button {
        background: var(--primary-color) !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInputContainer button:hover {
        background: var(--primary-hover) !important;
        transform: scale(1.05) !important;
    }

    .status-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--surface);
        z-index: 1000;
        border-bottom: 1px solid var(--border);
    }
    
    .status-active {
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--success-color));
        width: 100%;
        animation: pulse 2s infinite;
    }

    .loading-message {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1.5rem;
        background: var(--surface);
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        max-width: 200px;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        font-size: 0.9rem;
        animation: fadeIn 0.3s ease;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background: var(--text-secondary);
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }

    .sources-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    .source-item {
        background: white;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        font-size: 0.85rem;
    }
    
    .confidence-meter {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    .confidence-bar {
        width: 100px;
        height: 4px;
        background: #e9ecef;
        border-radius: 2px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: var(--success-color);
        transition: width 0.3s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }

    .footer {
        text-align: center;
        padding: 3rem 2rem 2rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border-top: 1px solid var(--border);
        margin-top: 4rem;
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .chat-container {
            padding: 0 1rem;
        }
        
        .stChatMessage[data-testid="user-message"] > div,
        .stChatMessage[data-testid="assistant-message"] > div {
            max-width: 90% !important;
        }
    }
    
    footer { display: none; }
    header[data-testid="stHeader"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# FUNCIONES DEL SISTEMA RAG
# ---------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = check_rag_system()
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

def check_rag_system():
    try:
        if not rag_system:
            return False
        health = rag_system.health_check()
        return health["status"] == "healthy"
    except Exception as e:
        logger.error(f"Error verificando sistema RAG: {e}")
        return False

def process_rag_query(user_query: str) -> None:
    if not user_query.strip():
        return
    
    if not st.session_state.rag_ready:
        st.error("Sistema RAG no disponible. Verifica la configuración.")
        return
    
    st.session_state.messages.append(("user", user_query))
    st.session_state.total_queries += 1
    
    typing_placeholder = st.empty()
    with typing_placeholder:
        st.markdown("""
            <div class="loading-message">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span>Buscando información...</span>
            </div>
        """, unsafe_allow_html=True)
    
    try:
        rag_response: RAGResponse = rag_system.query(user_query)
        typing_placeholder.empty()
        
        st.session_state.messages.append(("assistant", rag_response.answer, rag_response))
        logger.info(f"Consulta procesada en {rag_response.processing_time:.2f}s")
        
    except Exception as e:
        typing_placeholder.empty()
        logger.error(f"Error procesando consulta RAG: {e}")
        st.session_state.messages.append((
            "assistant", 
            "Lo siento, hubo un error procesando tu consulta. Por favor, intenta nuevamente.",
            None
        ))

def render_message_with_sources(role: str, message: str, rag_response: Optional[RAGResponse] = None):
    with st.chat_message(role):
        st.markdown(message)
        
        if role == "assistant" and rag_response and rag_response.sources:
            confidence_percent = int(rag_response.confidence * 100)
            confidence_color = "#28a745" if rag_response.confidence > 0.7 else "#ffc107" if rag_response.confidence > 0.4 else "#dc3545"
            
            st.markdown(f"""
                <div class="confidence-meter">
                    <span>Confianza:</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%; background: {confidence_color};"></div>
                    </div>
                    <span>{confidence_percent}%</span>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📚 Fuentes consultadas ({len(rag_response.sources)})"):
                for i, source in enumerate(rag_response.sources, 1):
                    st.markdown(f"""
                        <div class="source-item">
                            <strong>Fuente {i}</strong> (Similitud: {source['score']:.1%})<br>
                            {source['text']}
                            {f"<br><small>📁 {source['metadata'].get('source', 'N/A')}</small>" if source['metadata'] else ""}
                        </div>
                    """, unsafe_allow_html=True)

def setup_sample_data():
    try:
        stats = pinecone_client.get_index_stats()
        
        if stats.get('total_vectors', 0) == 0:
            with st.spinner("Configurando datos de ejemplo del TUPA..."):
                sample_docs = create_sample_tupa_documents()
                success = pinecone_client.upsert_documents(sample_docs)
                
                if success:
                    st.success("✅ Datos de ejemplo configurados exitosamente")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Error configurando datos de ejemplo")
        
    except Exception as e:
        st.error(f"Error configurando datos: {e}")

# ---------------------------
# UI PRINCIPAL
# ---------------------------
init_session()

st.markdown('<div class="status-bar"><div class="status-active"></div></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🤖 Sistema RAG TUPA")
    st.divider()
    
    st.subheader("📊 Estado del Sistema")
    
    if st.session_state.rag_ready:
        try:
            health = rag_system.health_check()
            pinecone_stats = pinecone_client.get_index_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RAG", "🟢 Activo")
            with col2:
                st.metric("Pinecone", "🟢 Conectado")
            
            st.metric("Documentos en BD", pinecone_stats.get('total_vectors', 0))
            st.metric("Modelo", rag_config.chat_model)
            
        except Exception as e:
            st.error(f"Error obteniendo estado: {e}")
    else:
        st.error("🔴 Sistema RAG no disponible")
    
    st.divider()
    
    st.subheader("📈 Estadísticas")
    st.metric("Consultas realizadas", st.session_state.total_queries)
    st.metric("Mensajes en chat", len(st.session_state.messages))
    
    st.divider()
    
    st.subheader("⚙️ Controles")
    
    if st.button("🔄 Nueva Conversación", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📚 Configurar Datos Ejemplo", use_container_width=True):
        setup_sample_data()

# Verificar RAG
if not st.session_state.rag_ready:
    st.error("❌ Sistema RAG no está disponible. Verifica tu configuración de Pinecone y OpenAI.")
    st.info("💡 Asegúrate de tener configuradas las variables: PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY")
    st.stop()

# Header cuando hay mensajes
if st.session_state.messages:
    st.markdown("""
        <div class="mini-header">
            <h3>🏛️ Asistente TUPA RAG</h3>
            <p>Sistema de Recuperación y Generación Aumentada</p>
        </div>
    """, unsafe_allow_html=True)

# Hero section
if not st.session_state.messages:
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Asistente TUPA RAG</h1>
            <p class="hero-subtitle">Gobierno Regional del Cusco</p>
            <p class="hero-description">
                Sistema inteligente con base de conocimientos para consultas sobre procedimientos administrativos
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Botones de consultas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Licencia de Funcionamiento", use_container_width=True):
            process_rag_query("¿Qué documentos necesito para obtener una licencia de funcionamiento?")
            st.rerun()
    
    with col2:
        if st.button("🏗️ Permisos de Construcción", use_container_width=True):
            process_rag_query("¿Cuánto tiempo demora el trámite de permiso de construcción?")
            st.rerun()
    
    with col3:
        if st.button("⏰ Horarios de Atención", use_container_width=True):
            process_rag_query("¿Cuáles son los horarios de atención de las oficinas?")
            st.rerun()
    
    with col4:
        if st.button("💰 Tasas y Costos", use_container_width=True):
            process_rag_query("¿Cuánto cuesta un certificado de zonificación?")
            st.rerun()

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Mostrar mensajes
for i, message_data in enumerate(st.session_state.messages):
    if len(message_data) == 3:
        role, message, rag_response = message_data
        render_message_with_sources(role, message, rag_response)
    else:
        role, message = message_data
        render_message_with_sources(role, message)

st.markdown('</div>', unsafe_allow_html=True)

# Input del chat
if prompt := st.chat_input("Pregunta sobre procedimientos del TUPA..."):
    process_rag_query(prompt)
    st.rerun()

# Footer
if st.session_state.messages:
    st.markdown("""
        <div class="footer">
            🏛️ Gobierno Regional del Cusco • Asistente TUPA RAG<br>
            <small>Powered by Pinecone + OpenAI</small>
        </div>
    """, unsafe_allow_html=True)
