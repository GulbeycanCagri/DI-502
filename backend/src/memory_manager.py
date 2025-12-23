"""
Memory Manager for Conversation History
Provides session-based chat memory using LlamaIndex's ChatMemoryBuffer.
Compatible with Llama models and supports async streaming.
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer


@dataclass
class SessionData:
    """Container for session-specific data."""

    memory: ChatMemoryBuffer
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class ConversationMemoryManager:
    """
    Manages conversation memory across multiple sessions.

    Features:
    - Session-based memory isolation
    - Automatic session cleanup for expired sessions
    - Thread-safe operations
    - Configurable token limit for memory buffer
    """

    def __init__(
        self,
        token_limit: int = 3000,
        session_timeout: int = 3600,  # 1 hour default
        max_sessions: int = 1000,
    ):
        """
        Initialize the memory manager.

        Args:
            token_limit: Maximum tokens to keep in memory buffer
            session_timeout: Session expiry time in seconds
            max_sessions: Maximum number of concurrent sessions
        """
        self._sessions: Dict[str, SessionData] = {}
        self._lock = Lock()
        self._token_limit = token_limit
        self._session_timeout = session_timeout
        self._max_sessions = max_sessions

    def get_or_create_session(self, session_id: str) -> ChatMemoryBuffer:
        """
        Get existing session memory or create a new one.

        Args:
            session_id: Unique identifier for the session

        Returns:
            ChatMemoryBuffer for the session
        """
        with self._lock:
            self._cleanup_expired_sessions()

            if session_id not in self._sessions:
                # Create new session with memory buffer
                memory = ChatMemoryBuffer.from_defaults(token_limit=self._token_limit)
                self._sessions[session_id] = SessionData(memory=memory)
                print(f"[Memory] Created new session: {session_id}")
            else:
                # Update last accessed time
                self._sessions[session_id].last_accessed = time.time()

            return self._sessions[session_id].memory

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the session's memory.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        memory = self.get_or_create_session(session_id)

        message_role = MessageRole.USER if role == "user" else MessageRole.ASSISTANT
        message = ChatMessage(role=message_role, content=content)
        memory.put(message)

        print(f"[Memory] Added {role} message to session {session_id}")

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get the chat history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of ChatMessage objects
        """
        memory = self.get_or_create_session(session_id)
        return memory.get_all()

    def get_history_as_string(self, session_id: str) -> str:
        """
        Get chat history formatted as a string for prompt injection.

        Args:
            session_id: Session identifier

        Returns:
            Formatted conversation history string
        """
        history = self.get_chat_history(session_id)

        if not history:
            return ""

        formatted_parts = []
        for msg in history:
            role_label = "User" if msg.role == MessageRole.USER else "Assistant"
            formatted_parts.append(f"{role_label}: {msg.content}")

        return "\n".join(formatted_parts)

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific session's memory.

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                print(f"[Memory] Cleared session: {session_id}")
                return True
            return False

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions based on timeout."""
        current_time = time.time()
        expired_sessions = [
            sid
            for sid, data in self._sessions.items()
            if current_time - data.last_accessed > self._session_timeout
        ]

        for sid in expired_sessions:
            del self._sessions[sid]
            print(f"[Memory] Expired session removed: {sid}")

        # If still over limit, remove oldest sessions
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(), key=lambda x: x[1].last_accessed
            )
            to_remove = len(self._sessions) - self._max_sessions
            for sid, _ in sorted_sessions[:to_remove]:
                del self._sessions[sid]
                print(f"[Memory] Session removed (limit reached): {sid}")

    def get_session_count(self) -> int:
        """Get the current number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return session_id in self._sessions


# Global instance for the application
memory_manager = ConversationMemoryManager(
    token_limit=20000,  # ~3000 tokens of history
    session_timeout=3600,  # 1 hour session timeout
    max_sessions=15,  # Max 1000 concurrent sessions
)
