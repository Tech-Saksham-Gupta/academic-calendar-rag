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

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  setTimeout(() => elements.toast.classList.remove('show'), 3000);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Something went wrong. Please try again.');
  return data;
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

async function uploadFiles(files) {
  const pdfs = Array.from(files).filter((file) => file.name.toLowerCase().endsWith('.pdf'));
  if (!pdfs.length) return showToast('Choose a PDF calendar to upload.');
  elements.progress.hidden = false;
  try {
    for (let index = 0; index < pdfs.length; index += 1) {
      const formData = new FormData();
      formData.append('file', pdfs[index]);
      elements.uploadMessage.textContent = `Indexing ${pdfs[index].name} (${index + 1} of ${pdfs.length})…`;
      const result = await request('/api/documents', { method: 'POST', body: formData });
      showToast(`${result.filename} added to your calendar library.`);
    }
    await refreshDocuments();
    document.getElementById('statusDot').className = 'status-dot ready';
    document.getElementById('statusText').textContent = 'Ready to help';
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.progress.hidden = true;
    elements.uploadMessage.textContent = '';
    elements.files.value = '';
    try { await refreshDocuments(); } catch (refreshError) { /* Keep the upload result visible. */ }
  }
}

function addMessage(role, text, sources = []) {
  elements.welcome.hidden = true;
  const message = document.createElement('div');
  message.className = `message ${role}`;
  const safeText = escapeHtml(text);
  if (role === 'user') {
    message.innerHTML = `<div class="message-body"><div class="message-label">YOU</div><div class="message-text">${safeText}</div></div>`;
  } else {
    const sourceCards = sources.length ? `<div class="source-grid">${sources.map((source, index) => `<div class="source-card"><div class="source-meta">SOURCE ${index + 1} · ${escapeHtml(source.filename)} · PAGE ${source.page}</div><div class="source-text">${escapeHtml(source.text)}</div></div>`).join('')}</div>` : '';
    message.innerHTML = `<div class="message-avatar">✳</div><div class="message-body"><div class="message-label">CALENDAR ASSISTANT</div><div class="message-text">${safeText}</div>${sourceCards}</div>`;
  }
  elements.messages.appendChild(message);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  return message;
}

async function askQuestion(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion) return;
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
      body: JSON.stringify({ question: cleanQuestion }),
    });
    waiting.remove();
    addMessage('assistant', result.answer, result.sources);
  } catch (error) {
    waiting.remove();
    addMessage('assistant', error.message);
  } finally {
    elements.send.disabled = false;
    elements.input.focus();
  }
}

elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  askQuestion(elements.input.value);
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
