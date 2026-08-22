(function () {
  "use strict";
  if (window.__atsThemeToggleLoaded) return;
  window.__atsThemeToggleLoaded = true;
  var KEY = "site-theme-pref";
  var MODES = ["default", "obs-night", "obs-day"];
  var LABELS = {
    "default": "Theme: site default",
    "obs-night": "Observatory \u00b7 night",
    "obs-day": "Observatory \u00b7 day"
  };
  function stored() {
    try {
      return localStorage.getItem(KEY);
    } catch (err) {
      return null;
    }
  }
  function current() {
    if (!document.body.classList.contains("theme-obs")) return "default";
    return document.body.getAttribute("data-mode") === "day" ? "obs-day" : "obs-night";
  }
  function apply(mode, save) {
    document.body.classList.toggle("theme-obs", mode !== "default");
    if (mode === "obs-day") {
      document.body.setAttribute("data-mode", "day");
    } else {
      document.body.removeAttribute("data-mode");
    }
    if (save) {
      try {
        localStorage.setItem(KEY, mode);
      } catch (err) {}
    }
    var button = document.getElementById("theme-toggle");
    if (button) button.textContent = LABELS[mode];
  }
  function cycle() {
    apply(MODES[(MODES.indexOf(current()) + 1) % MODES.length], true);
  }
  function mount() {
    var host = document.getElementById("theme-toggle-mount");
    if (!host) host = document.body;
    var button = document.createElement("button");
    button.type = "button";
    button.id = "theme-toggle";
    button.className = "theme-toggle-button";
    button.textContent = LABELS.default;
    button.addEventListener("click", cycle);
    host.appendChild(button);
    var saved = stored();
    apply(MODES.indexOf(saved) === -1 ? "default" : saved, false);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
