const elements = {
  conversation: document.getElementById('conversation'),
  welcome: document.getElementById('welcome'),
  messages: document.getElementById('messages'),
  input: document.getElementById('questionInput'),
  form: document.getElementById('chatForm'),
  send: document.querySelector('.send-button'),
  files: document.getElementById('fileInput'),
  dropzone: document.getElementById('dropzone'),
  library: document.getElementById('libraryList'),
  sidebarDocs: document.getElementById('documentList'),
  progress: document.getElementById('uploadProgress'),
  uploadMessage: document.getElementById('uploadMessage'),
  toast: document.getElementById('toast'),
};

let toastTimer;
let uploadInProgress = false;
let questionInProgress = false;
let conversationHistory = [];

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove('show'), 4000);
}

async function request(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch (error) { /* Use a friendly error below. */ }
    if (!response.ok) throw new Error(data.detail || 'The service could not complete that request. Please try again.');
    return data;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('This is taking longer than expected. The request was stopped; please try again with a smaller PDF.');
    }
    if (error instanceof TypeError) {
      throw new Error('The website could not reach its server. Refresh the page and try again.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function refreshDocuments() {
  const { documents } = await request('/api/documents');
  document.getElementById('documentCount').textContent = documents.length;
  document.getElementById('indexedCount').textContent = documents.length;

  elements.sidebarDocs.innerHTML = documents.length
    ? documents.map((doc) => `<div class="sidebar-doc"><span class="pdf-icon">PDF</span><span class="sidebar-doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span></div>`).join('')
    : '<div class="empty-docs">Your calendars will appear here.</div>';

  elements.library.innerHTML = documents.length
    ? documents.map((doc) => `<div class="library-document"><span class="pdf-icon">PDF</span><div class="library-info"><strong title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</strong><small>${doc.passages} searchable passages</small></div><button class="remove-doc" data-id="${doc.id}" title="Remove calendar" aria-label="Remove ${escapeHtml(doc.filename)}">×</button></div>`).join('')
    : '<div class="library-empty"><span>▤</span><strong>No calendars yet</strong><small>Your uploaded PDFs will show up here.</small></div>';
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

function formatAnswer(text) {
  const safe = escapeHtml(text);
  const months = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec';
  const datePattern = new RegExp(`\\b(?:${months})\\s+\\d{1,2}(?:st|nd|rd|th)?(?:\\s*[-–—]\\s*\\d{1,2})?(?:,?\\s+\\d{4})?|\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b`, 'gi');
  return safe.replace(datePattern, '<strong>$&</strong>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

async function uploadFiles(files) {
  if (uploadInProgress) return;
  const pdfs = Array.from(files).filter((file) => file.name.toLowerCase().endsWith('.pdf'));
  if (!pdfs.length) return showToast('Choose a PDF calendar to upload.');
  uploadInProgress = true;
  elements.files.disabled = true;
  elements.progress.hidden = false;
  try {
    for (let index = 0; index < pdfs.length; index += 1) {
      const formData = new FormData();
      formData.append('file', pdfs[index]);
      elements.uploadMessage.textContent = `Indexing ${pdfs[index].name} (${index + 1} of ${pdfs.length})…`;
      const result = await request('/api/documents', { method: 'POST', body: formData });
      showToast(`${result.filename} added to your calendar library.`);
    }
    document.getElementById('statusDot').className = 'status-dot ready';
    document.getElementById('statusText').textContent = 'Ready to help';
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.progress.hidden = true;
    elements.uploadMessage.textContent = '';
    elements.files.value = '';
    elements.files.disabled = false;
    uploadInProgress = false;
    try { await refreshDocuments(); } catch (refreshError) { /* Keep the upload result visible. */ }
  }
}

function addMessage(role, text, sources = [], suggestions = []) {
  elements.welcome.hidden = true;
  const message = document.createElement('div');
  message.className = `message ${role}`;
  const safeText = formatAnswer(text);
  if (role === 'user') {
    message.innerHTML = `<div class="message-body"><div class="message-label">YOU</div><div class="message-text">${safeText}</div></div>`;
  } else {
    message.innerHTML = `<div class="message-avatar">✳</div><div class="message-body"><div class="message-label">CALENDAR ASSISTANT</div><div class="message-text">${safeText}</div></div>`;
    const body = message.querySelector('.message-body');
    if (Array.isArray(suggestions) && suggestions.length) {
      const followUps = document.createElement('div');
      followUps.className = 'follow-up-list';
      const label = document.createElement('div');
      label.className = 'follow-up-label';
      label.textContent = 'YOU COULD ALSO ASK';
      followUps.appendChild(label);
      suggestions.forEach((question) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'follow-up-question';
        button.dataset.question = question;
        button.textContent = question;
        followUps.appendChild(button);
      });
      body.appendChild(followUps);
    }
  }
  elements.messages.appendChild(message);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  return message;
}

async function askQuestion(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion || questionInProgress) return;
  questionInProgress = true;
  elements.input.value = '';
  elements.input.style.height = 'auto';
  addMessage('user', cleanQuestion);
  elements.send.disabled = true;
  const waiting = document.createElement('div');
  waiting.className = 'message';
  waiting.innerHTML = '<div class="message-avatar">✳</div><div class="thinking">Searching your calendar <span class="dots"><span>●</span> <span>●</span> <span>●</span></span></div>';
  elements.messages.appendChild(waiting);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  try {
    const result = await request('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: cleanQuestion, history: conversationHistory.slice(-8) }),
    });
    waiting.remove();
    addMessage('assistant', result.answer, result.sources, result.suggestions);
    conversationHistory.push({ role: 'user', content: cleanQuestion.slice(0, 1_500) });
    conversationHistory.push({ role: 'assistant', content: String(result.answer).slice(0, 1_500) });
  } catch (error) {
    waiting.remove();
    addMessage('assistant', error.message);
  } finally {
    questionInProgress = false;
    elements.send.disabled = false;
    elements.input.focus();
  }
}

elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  askQuestion(elements.input.value);
});
elements.messages.addEventListener('click', (event) => {
  const button = event.target.closest('.follow-up-question');
  if (button) askQuestion(button.dataset.question);
});
elements.input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.input.addEventListener('input', () => {
  elements.input.style.height = 'auto';
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 100)}px`;
});
document.querySelectorAll('.suggestion').forEach((button) => button.addEventListener('click', () => {
  elements.input.value = button.dataset.question;
  askQuestion(button.dataset.question);
}));
elements.files.addEventListener('change', () => uploadFiles(elements.files.files));
elements.dropzone.addEventListener('dragover', (event) => { event.preventDefault(); elements.dropzone.classList.add('dragging'); });
elements.dropzone.addEventListener('dragleave', () => elements.dropzone.classList.remove('dragging'));
elements.dropzone.addEventListener('drop', (event) => { event.preventDefault(); elements.dropzone.classList.remove('dragging'); uploadFiles(event.dataTransfer.files); });
elements.library.addEventListener('click', async (event) => {
  const button = event.target.closest('.remove-doc');
  if (!button) return;
  try {
    await request(`/api/documents/${button.dataset.id}`, { method: 'DELETE' });
    await refreshDocuments();
    showToast('Calendar removed from your library.');
  } catch (error) { showToast(error.message); }
});
document.getElementById('clearChatButton').addEventListener('click', () => {
  elements.messages.innerHTML = '';
  conversationHistory = [];
  elements.welcome.hidden = false;
});
document.getElementById('newChatButton').addEventListener('click', () => document.getElementById('clearChatButton').click());
document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault(); elements.input.focus();
  }
});

async function start() {
  try {
    const health = await request('/api/health');
    const dot = document.getElementById('statusDot');
    dot.className = health.ready ? 'status-dot ready' : 'status-dot error';
    document.getElementById('statusText').textContent = health.message;
    await refreshDocuments();
  } catch (error) {
    document.getElementById('statusText').textContent = 'Connection issue';
    showToast(error.message);
  }
}

start();
