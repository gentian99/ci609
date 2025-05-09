//sidebar toggling
function openSideBar() {
  document.getElementById("sidebar").style.width = "250px";
}
function closeSideBar() {
  document.getElementById("sidebar").style.width = "0";
}

//video playback speed
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('video source[src*="video1.mp4"]').forEach(source =>
    source.parentElement.playbackRate = 0.6
  );
});

//typewriter output
function typeMessageToElement(elemId, message, color = '#00FF00') {
  const el = document.getElementById(elemId);
  if (!el) return;

  el.textContent = "";
  el.style.color = color;
  el.style.fontFamily = "'Bebas Neue', sans-serif";
  el.style.fontSize = "20px";
  el.style.whiteSpace = "pre-wrap";

  let idx = 0;
  function typeChar() {
    if (idx < message.length) {
      el.textContent += message[idx++];
      setTimeout(typeChar, 30);
    }
  }
  typeChar();
}

document.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('predict-layout');
  const loggedIn = root ? root.dataset.loggedIn === 'true' : false;
  const outputElem = document.getElementById('console-output');

  const loginForm = document.getElementById('login-form');
  const signupForm = document.getElementById('signup-form');
  const form = document.getElementById('predict-form');
  const homeSelect = document.getElementById('home_team');
  const awaySelect = document.getElementById('away_team');
  const leagueSelect = document.getElementById('league');
  const saveBtn = document.getElementById('save-btn');
  const exportBtn = document.getElementById('export-btn');
  const clearBtn = document.getElementById('clear-btn');

  let savedEntries = JSON.parse(localStorage.getItem('savedConsoleEntries') || '[]');
  let currentLine = "";

  //unified auth form handling ---
  async function handleFormSubmit(form, endpoint, redirectUrl, creating = false) {
    const data = Object.fromEntries(new FormData(form).entries());
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const json = await res.json();
      if (res.ok) {
        typeMessageToElement('form-message', creating ? 'Creating account...' : 'Logging in...', '#00FF00');
        await new Promise(res => setTimeout(res, 600));
        typeMessageToElement('form-message', creating ? 'Account created. Redirecting...' : 'Login successful. Redirecting...', '#00FF00');
        setTimeout(() => window.location.href = redirectUrl, 800);
      } else {
        typeMessageToElement('form-message', json.error || 'Error occurred.', 'crimson');
      }
    } catch {
      typeMessageToElement('form-message', 'Network error.', 'crimson');
    }
  }

  if (loginForm) loginForm.addEventListener('submit', e => {
    e.preventDefault();
    handleFormSubmit(loginForm, '/api/login', '/predict');
  });

  if (signupForm) signupForm.addEventListener('submit', e => {
    e.preventDefault();
    handleFormSubmit(signupForm, '/api/signup', '/login', true);
  });

  //disable same team in home/away selects
  function updateOptions() {
    if (!homeSelect || !awaySelect) return;
    const h = homeSelect.value, a = awaySelect.value;
    [...awaySelect.options].forEach(opt => opt.disabled = (opt.value === h));
    [...homeSelect.options].forEach(opt => opt.disabled = (opt.value === a));
  }

  if (homeSelect && awaySelect) {
    homeSelect.addEventListener('change', updateOptions);
    awaySelect.addEventListener('change', updateOptions);
    updateOptions();
  }

  // update team list on league change
  if (leagueSelect) {
    leagueSelect.addEventListener('change', async () => {
      try {
        const league = leagueSelect.value;
        const res = await fetch(`/api/teams?league=${league}`);
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Failed to load teams');

        homeSelect.innerHTML = "";
        awaySelect.innerHTML = "";
        json.teams.forEach(team => {
          homeSelect.add(new Option(team, team));
          awaySelect.add(new Option(team, team));
        });

        updateOptions();
      } catch {
        typeCurrentLine("Error: Failed to fetch teams for selected league");
      }
    });
  }

  //console typewriter logic ---
  function renderConsole() {
    if (!outputElem) return;
    outputElem.style.color = currentLine.trim().toLowerCase().startsWith("error:") ? 'crimson' : '#00FF00';
    outputElem.textContent = (savedEntries.join("\n\n") || "") + (savedEntries.length ? "\n\n" : "") + currentLine;
  }

  function typeCurrentLine(line) {
    currentLine = "";
    renderConsole();
    let idx = 0;
    function typeChar() {
      if (idx < line.length) {
        currentLine += line[idx++];
        renderConsole();
        setTimeout(typeChar, 30);
      }
    }
    typeChar();
  }

  //console buttons
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      if (!loggedIn) return typeCurrentLine("Error: Please log in to save results.");
      if (currentLine) {
        savedEntries.push(currentLine);
        localStorage.setItem('savedConsoleEntries', JSON.stringify(savedEntries));
        currentLine = "";
        renderConsole();
      }
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!loggedIn) return typeCurrentLine("Error: Please log in to export results.");
      const blob = new Blob([savedEntries.join("\n")], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "predictions.txt";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      savedEntries = [];
      localStorage.setItem('savedConsoleEntries', '[]');
      currentLine = "";
      renderConsole();
    });
  }

  //submit prediction request
  if (form) {
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const home = homeSelect.value, away = awaySelect.value;
      const league = leagueSelect ? leagueSelect.value : '2024_2025';

      if (home === away) return typeCurrentLine("Error: Home and away teams must be different");

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ home_team: home, away_team: away, league })
        });
        const data = await res.json();
        if (!res.ok) return typeCurrentLine(`Error: ${data.error}`);
        typeCurrentLine(`[${data.timestamp}] > ${data.home_team} vs ${data.away_team} → ${data.result}`);
      } catch {
        typeCurrentLine("Error: Network error");
      }
    });
  }

  renderConsole();
});
