import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

// Firebase Config
const firebaseConfig = {
    apiKey: "AIzaSyD4iEMkXq3gJljeyF4wmF8UYAaqzDltSEA",
    authDomain: "gen-lang-client-0781062109.firebaseapp.com",
    projectId: "gen-lang-client-0781062109",
    storageBucket: "gen-lang-client-0781062109.firebasestorage.app",
    messagingSenderId: "638445321934",
    appId: "1:638445321934:web:43a5c267d4b67f37ab1a9f",
    measurementId: "G-SCYRJNF96C"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

let currentSessionId = null;
let currentUser = null;
let isGenerating = false;

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');
    const newChatBtn = document.getElementById('newChatBtn');
    const sourceModal = document.getElementById('sourceModal');
    const closeBtn = document.querySelector('.close-btn');
    const sourceDetails = document.getElementById('sourceDetails');
    const logoutBtn = document.getElementById('logoutBtn');
    const userEmailSpan = document.getElementById('userEmail');
    const chatHistoryList = document.getElementById('chatHistoryList');

    // Authentication Guard
    onAuthStateChanged(auth, (user) => {
        if (user) {
            currentUser = user;
            userEmailSpan.innerText = user.email;
            loadChatHistory();
        } else {
            window.location.href = "/static/login.html";
        }
    });

    logoutBtn.addEventListener('click', () => {
        signOut(auth).then(() => {
            window.location.href = "/static/login.html";
        });
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        if (isGenerating) return;
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim().length > 0) {
            sendBtn.disabled = false;
        } else {
            sendBtn.disabled = true;
        }
    });

    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && !isGenerating) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    newChatBtn.addEventListener('click', () => {
        if (isGenerating) return; // Chặn tạo chat mới khi đang chờ Bot
        startNewChat();
    });

    function startNewChat() {
        currentSessionId = null;
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="message-content">
                    <p>Xin chào! Tôi là trợ lý AI chuyên về môn Sinh học lớp 12. Tôi có thể giúp bạn tìm kiếm thông tin, giải thích các khái niệm, cơ chế hay quá trình sinh học dựa trên kiến thức trong SGK. Bạn cần hỏi gì nào?</p>
                </div>
            </div>
        `;
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;
        userInput.focus();
    }

    closeBtn.onclick = function() {
        sourceModal.style.display = "none";
    }
    window.onclick = function(event) {
        if (event.target == sourceModal) {
            sourceModal.style.display = "none";
        }
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid fa-user"></i></div>
            <div class="message-content">
                <p>${text.replace(/\n/g, '<br>')}</p>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendLoadingMessage() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message loading-message';
        msgDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function appendBotMessage(text, sources) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = '<div class="sources-container">';
            sourcesHtml += `<button class="source-btn" data-sources='${JSON.stringify(sources).replace(/'/g, "&#39;")}'>
                                <i class="fa-solid fa-book-open"></i> Xem ${sources.length} nguồn tài liệu
                            </button>`;
            sourcesHtml += '</div>';
        }

        const renderedHtml = marked.parse(text);

        msgDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="message-content">
                ${renderedHtml}
                ${sourcesHtml}
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        
        const btn = msgDiv.querySelector('.source-btn');
        if (btn) {
            btn.addEventListener('click', function() {
                const sourceData = JSON.parse(this.getAttribute('data-sources'));
                showSources(sourceData);
            });
        }
        
        scrollToBottom();
    }

    function showSources(sources) {
        let html = '';
        sources.forEach((src, index) => {
            let tagsHtml = '';
            if (src.khai_niem && src.khai_niem.length > 0) {
                src.khai_niem.forEach(kn => {
                    tagsHtml += `<span class="tag">${kn}</span>`;
                });
            }

            const markdownContent = marked.parse(src.noi_dung);
            let sourceTitle = `Nguồn ${index + 1}: ${src.bai_hoc}`;
            if (src.tieu_muc && src.tieu_muc !== "None") {
                sourceTitle += ` - ${src.tieu_muc}`;
            }
            if (src.trang && src.trang.length > 0) {
                sourceTitle += ` (Trang ${src.trang.join(', ')})`;
            }

            html += `
                <div class="source-item">
                    <h6 class="text-primary mb-2">${sourceTitle} (Score: ${(src.score * 100).toFixed(1)}%)</h6>
                    <div class="text-light-50 mb-0 source-markdown-content" style="font-size: 0.9rem;">${markdownContent}</div>
                    ${tagsHtml ? '<div style="margin-top:0.5rem"><strong>Khái niệm:</strong> ' + tagsHtml + '</div>' : ''}
                </div>
            `;
        });
        sourceDetails.innerHTML = html;
        sourceModal.style.display = "flex";
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function loadChatHistory() {
        if (!currentUser) return;
        try {
            const token = await currentUser.getIdToken();
            const res = await fetch('/history', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const histories = await res.json();
                renderHistoryList(histories);
            }
        } catch (error) {
            console.error("Error loading history", error);
        }
    }

    function renderHistoryList(histories) {
        chatHistoryList.innerHTML = '';
        histories.forEach(h => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.style.cssText = 'padding: 0.75rem; border-radius: 6px; cursor: pointer; color: #ececf1; display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; transition: background 0.2s;';
            div.onmouseover = () => div.style.background = '#2A2B32';
            div.onmouseout = () => div.style.background = 'transparent';
            
            const titleSpan = document.createElement('span');
            titleSpan.innerText = h.title || "Đoạn chat mới";
            titleSpan.style.cssText = 'overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 150px; font-size: 0.9rem;';
            titleSpan.onclick = () => {
                if (isGenerating) return; // Chặn load lịch sử khi đang chờ Bot
                loadSession(h.session_id);
            };
            
            const delBtn = document.createElement('i');
            delBtn.className = 'fa-regular fa-trash-can';
            delBtn.style.cssText = 'color: #8e8ea0; cursor: pointer; padding: 5px;';
            delBtn.onmouseover = () => delBtn.style.color = '#ef4444';
            delBtn.onmouseout = () => delBtn.style.color = '#8e8ea0';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                if (isGenerating) return; // Chặn xóa khi đang chờ Bot
                deleteSession(h.session_id);
            };

            div.appendChild(titleSpan);
            div.appendChild(delBtn);
            chatHistoryList.appendChild(div);
        });
    }

    async function loadSession(sessionId) {
        if (!currentUser) return;
        currentSessionId = sessionId;
        try {
            const token = await currentUser.getIdToken();
            const res = await fetch(`/history/${sessionId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                chatMessages.innerHTML = ''; // Clear current messages
                data.messages.forEach(msg => {
                    if (msg.role === 'user') appendUserMessage(msg.content);
                    else appendBotMessage(msg.content, msg.sources);
                });
            }
        } catch (error) {
            console.error("Error loading session", error);
        }
    }

    async function deleteSession(sessionId) {
        if (!confirm("Bạn có chắc muốn xóa đoạn chat này?")) return;
        try {
            const token = await currentUser.getIdToken();
            const res = await fetch(`/history/${sessionId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                if (currentSessionId === sessionId) {
                    startNewChat();
                }
                loadChatHistory();
            }
        } catch (error) {
            console.error("Error deleting session", error);
        }
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isGenerating || !currentUser) return;
        
        const text = userInput.value.trim();
        if (!text) return;

        appendUserMessage(text);
        
        isGenerating = true;
        userInput.value = '';
        userInput.style.height = 'auto';
        userInput.disabled = true;
        userInput.placeholder = "Đang chờ Bot trả lời...";
        sendBtn.disabled = true;

        const loadingMsg = appendLoadingMessage();

        try {
            const token = await currentUser.getIdToken();
            const payload = { query: text };
            if (currentSessionId) {
                payload.session_id = currentSessionId;
            }

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(payload)
            });

            if (response.status === 401) {
                const errorData = await response.json();
                if (errorData.detail === "TOKEN_REVOKED" || errorData.detail === "USER_DISABLED" || errorData.detail.includes("Invalid")) {
                    alert("Tài khoản của bạn đã bị khóa hoặc phiên đăng nhập hết hạn. Bạn sẽ bị đăng xuất.");
                    await signOut(auth);
                    window.location.href = "/static/login.html";
                    return;
                }
            } else if (response.status === 429) {
                const errorData = await response.json();
                alert(errorData.detail);
                throw new Error("Spam detected");
            } else if (!response.ok) {
                throw new Error('Lỗi từ server');
            }

            const data = await response.json();
            
            if (data.session_id) {
                currentSessionId = data.session_id;
                loadChatHistory(); // Refresh history list
            }

            chatMessages.removeChild(loadingMsg);
            appendBotMessage(data.answer, data.sources);

        } catch (error) {
            console.error('Error:', error);
            chatMessages.removeChild(loadingMsg);
            appendBotMessage('Xin lỗi, đã có lỗi xảy ra khi kết nối tới hệ thống. Vui lòng thử lại sau.');
        } finally {
            isGenerating = false;
            userInput.disabled = false;
            userInput.placeholder = "Hỏi tôi bất cứ điều gì về Sinh học 12...";
            setTimeout(() => userInput.focus(), 100);
        }
    });
});

document.addEventListener('click', function(e) {
    if (e.target.tagName === 'IMG' && (e.target.closest('.message-content') || e.target.closest('.source-item'))) {
        const lightbox = document.getElementById('imageLightbox');
        const lightboxImg = document.getElementById('lightboxImg');
        lightboxImg.src = e.target.src;
        lightbox.style.display = 'flex';
    }
});