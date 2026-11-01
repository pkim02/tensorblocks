/* UI kit · kit.js
   Only what HTML can't do yet: tabs, toasts, anchoring menus, and fallbacks
   for browsers without invoker commands, anchor positioning or closedby.
   Load with <script src="kit.js" defer></script>. Exposes window.kit.toast(). */
(() => {
  const supportsCommand = "command" in HTMLButtonElement.prototype;
  const supportsAnchor = CSS.supports("anchor-name: --a");
  const supportsClosedBy = "closedBy" in HTMLDialogElement.prototype;

  /* ── Invoker commands fallback (command / commandfor) ── */
  if (!supportsCommand) {
    document.addEventListener("click", (e) => {
      const btn = e.target.closest("button[commandfor]");
      if (!btn) return;
      const target = document.getElementById(btn.getAttribute("commandfor"));
      if (!target) return;
      const cmd = btn.getAttribute("command");
      if (cmd === "show-modal") target.showModal();
      else if (cmd === "close" || cmd === "request-close") target.close(btn.value || undefined);
      else if (cmd === "toggle-popover") target.togglePopover();
      else if (cmd === "show-popover") target.showPopover();
      else if (cmd === "hide-popover") target.hidePopover();
    });
  }

  /* ── Dialog: click outside closes when closedby="any" ── */
  if (!supportsClosedBy) {
    document.addEventListener("click", (e) => {
      const d = e.target;
      if (!(d instanceof HTMLDialogElement) || !d.open || d.getAttribute("closedby") !== "any") return;
      const r = d.getBoundingClientRect();
      const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
      if (!inside) d.close();
    });
  }

  /* ── Menus: anchor to their button, arrow keys, close on pick ── */
  let anchorCount = 0;
  const invokerFor = (id) => document.querySelector(`[commandfor="${id}"], [popovertarget="${id}"]`);

  function wireMenu(menu) {
    if (menu.dataset.kitWired) return;
    menu.dataset.kitWired = "1";
    const invoker = menu.id && invokerFor(menu.id);
    if (invoker && supportsAnchor) {
      const name = `--kit-anchor-${++anchorCount}`;
      invoker.style.anchorName = name;
      menu.style.positionAnchor = name;
    }
    if (invoker) {
      invoker.setAttribute("aria-haspopup", "menu");
      invoker.setAttribute("aria-expanded", "false");
    }
    menu.addEventListener("toggle", (e) => {
      const open = e.newState === "open";
      if (invoker) invoker.setAttribute("aria-expanded", String(open));
      if (!open) return;
      if (invoker && !supportsAnchor) placeBelow(menu, invoker);
      items(menu)[0]?.focus();
    });
    menu.addEventListener("keydown", (e) => {
      const list = items(menu);
      const i = list.indexOf(document.activeElement);
      const go = (n) => { e.preventDefault(); list[(n + list.length) % list.length]?.focus(); };
      if (e.key === "ArrowDown") go(i + 1);
      else if (e.key === "ArrowUp") go(i - 1);
      else if (e.key === "Home") go(0);
      else if (e.key === "End") go(list.length - 1);
    });
    menu.addEventListener("click", (e) => {
      const item = e.target.closest("button, a");
      if (item && item.parentElement === menu && !item.hasAttribute("data-keep-open")) menu.hidePopover();
    });
  }
  const items = (menu) => [...menu.querySelectorAll(":scope > button:not(:disabled), :scope > a[href]")];

  function placeBelow(menu, invoker) {
    const r = invoker.getBoundingClientRect();
    const m = menu.getBoundingClientRect();
    const below = r.bottom + 4 + m.height <= innerHeight;
    menu.style.position = "fixed";
    menu.style.top = `${below ? r.bottom + 4 : Math.max(8, r.top - 4 - m.height)}px`;
    menu.style.left = `${Math.min(r.left, innerWidth - m.width - 8)}px`;
    menu.style.margin = "0";
  }

  /* ── Tabs: .tabs > [role=tablist] > [role=tab][aria-controls] ── */
  function wireTabs(root) {
    if (root.dataset.kitWired) return;
    root.dataset.kitWired = "1";
    const tabs = [...root.querySelectorAll('[role="tab"]')];
    const select = (tab, focus) => {
      for (const t of tabs) {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        const panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      }
      if (focus) tab.focus();
    };
    for (const t of tabs) {
      const panel = document.getElementById(t.getAttribute("aria-controls"));
      if (panel) {
        panel.setAttribute("role", "tabpanel");
        panel.tabIndex = 0;
        if (t.id) panel.setAttribute("aria-labelledby", t.id);
      }
      t.addEventListener("click", () => select(t, false));
      t.addEventListener("keydown", (e) => {
        const i = tabs.indexOf(t);
        const go = (n) => { e.preventDefault(); select(tabs[(n + tabs.length) % tabs.length], true); };
        if (e.key === "ArrowRight") go(i + 1);
        else if (e.key === "ArrowLeft") go(i - 1);
        else if (e.key === "Home") go(0);
        else if (e.key === "End") go(tabs.length - 1);
      });
    }
    select(tabs.find((t) => t.getAttribute("aria-selected") === "true") || tabs[0], false);
  }

  /* ── Chips: data-max limits how many can be picked ── */
  document.addEventListener("change", (e) => {
    const group = e.target.closest(".chips[data-max]");
    if (!group) return;
    const boxes = [...group.querySelectorAll('input[type="checkbox"]')];
    const full = boxes.filter((b) => b.checked).length >= +group.dataset.max;
    for (const b of boxes) if (!b.checked) b.disabled = full;
  });

  /* ── Toasts ── */
  let region;
  function toast(message, { tone = "", action, onAction, duration = 4000 } = {}) {
    if (!region) {
      region = document.createElement("div");
      region.className = "toasts";
      region.setAttribute("popover", "manual");
      region.setAttribute("role", "status");
      region.setAttribute("aria-live", "polite");
      document.body.append(region);
    }
    // Re-show so the region stays above any dialog opened since.
    if (region.matches(":popover-open")) region.hidePopover();
    region.showPopover();

    const el = document.createElement("div");
    el.className = `toast ${tone}`.trim();
    const text = document.createElement("span");
    text.className = "grow";
    text.textContent = message;
    el.append(text);
    const dismiss = () => {
      el.classList.add("leaving");
      setTimeout(() => {
        el.remove();
        if (!region.children.length && region.matches(":popover-open")) region.hidePopover();
      }, 200);
    };
    if (action) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = action;
      b.addEventListener("click", () => { onAction?.(); dismiss(); });
      el.append(b);
    }
    region.append(el);
    if (duration) setTimeout(dismiss, duration);
    return dismiss;
  }

  /* ── Boot, and watch for markup added later ── */
  function scan(root = document) {
    root.querySelectorAll(".menu[popover]").forEach(wireMenu);
    root.querySelectorAll(".tabs").forEach(wireTabs);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => scan());
  else scan();
  new MutationObserver(() => scan()).observe(document.documentElement, { childList: true, subtree: true });

  window.kit = { toast, scan };
})();
