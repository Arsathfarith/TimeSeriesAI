window.White = {
  toast(message) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3800);
  },
  highlight(element) {
    if (!element) return;
    element.style.transition = 'background 0.3s ease';
    element.style.background = 'rgba(55, 216, 240, 0.12)';
    setTimeout(() => element.style.background = '', 600);
  }
};
