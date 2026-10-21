// Pay-as-you-go rates (NZD). Keep in sync with the pricing cards in index.html.
const PRICING = { firstHour: 3, perExtraHour: 2, dailyCap: 10 };

// Placeholder stations: replace with your real venues (or load from an API).
const STATIONS = [
  { name: "Ponsonby Central", suburb: "Ponsonby", type: "Café", hours: "7am – 10pm", available: 6, slots: 2 },
  { name: "Britomart Transport Centre", suburb: "Auckland CBD", type: "Transport", hours: "24 hours", available: 9, slots: 3 },
  { name: "University Student Hub", suburb: "Grafton", type: "Campus", hours: "7am – 11pm", available: 2, slots: 6 },
  { name: "Viaduct Harbour Bar", suburb: "Viaduct", type: "Bar", hours: "11am – 2am", available: 4, slots: 4 },
  { name: "K Road Coffee Co.", suburb: "Karangahape Rd", type: "Café", hours: "6:30am – 4pm", available: 0, slots: 8 },
  { name: "Newmarket Train Station", suburb: "Newmarket", type: "Transport", hours: "5am – 1am", available: 7, slots: 1 },
  { name: "Wynyard Quarter Taphouse", suburb: "Wynyard Quarter", type: "Bar", hours: "12pm – 12am", available: 5, slots: 3 },
  { name: "AUT City Library", suburb: "Auckland CBD", type: "Campus", hours: "8am – 10pm", available: 3, slots: 5 },
];

const money = n => "$" + (Number.isInteger(n) ? n : n.toFixed(2));

// Mobile nav
const toggle = document.querySelector(".nav__toggle");
const links = document.getElementById("nav-links");
toggle.addEventListener("click", () => {
  const open = links.classList.toggle("is-open");
  toggle.setAttribute("aria-expanded", open);
});
links.addEventListener("click", e => {
  if (e.target.closest("a")) {
    links.classList.remove("is-open");
    toggle.setAttribute("aria-expanded", "false");
  }
});

// Pricing labels + cost calculator
document.querySelector('[data-price="first"]').textContent = money(PRICING.firstHour);
document.querySelector('[data-price="hourly"]').textContent = money(PRICING.perExtraHour);
document.querySelector('[data-price="daily"]').textContent = money(PRICING.dailyCap);

const hours = document.getElementById("hours");
const hoursOut = document.getElementById("hours-out");
const costOut = document.getElementById("cost-out");
function updateCost() {
  const h = Number(hours.value);
  const cost = Math.min(PRICING.firstHour + (h - 1) * PRICING.perExtraHour, PRICING.dailyCap);
  hoursOut.textContent = h === 1 ? "1 hour" : `${h} hours`;
  costOut.textContent = money(cost);
}
hours.addEventListener("input", updateCost);
updateCost();

// Station finder
const list = document.getElementById("station-list");
const empty = document.getElementById("station-empty");
const search = document.getElementById("station-search");
const chips = document.querySelectorAll(".chip");
let filter = "all";

function availability(n) {
  if (n === 0) return `<span><i class="dot dot--none"></i>None available</span>`;
  return `<span><i class="dot ${n <= 2 ? "dot--low" : ""}"></i>${n} to rent</span>`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderStations() {
  const q = search.value.trim().toLowerCase();
  const matches = STATIONS.filter(s =>
    (filter === "all" || s.type === filter) &&
    (!q || `${s.name} ${s.suburb}`.toLowerCase().includes(q))
  );
  list.innerHTML = matches.map(s => `
    <li class="station">
      <div class="station__top">
        <h3>${escapeHtml(s.name)}</h3>
        <span class="station__type">${escapeHtml(s.type)}</span>
      </div>
      <p class="station__meta">${escapeHtml(s.suburb)} · ${escapeHtml(s.hours)}</p>
      <div class="station__avail">
        ${availability(s.available)}
        <span class="muted">${s.slots} free slots</span>
      </div>
    </li>`).join("");
  empty.hidden = matches.length > 0;
}

chips.forEach(chip => chip.addEventListener("click", () => {
  chips.forEach(c => c.classList.toggle("is-active", c === chip));
  filter = chip.dataset.filter;
  renderStations();
}));
search.addEventListener("input", renderStations);
renderStations();

// Contact form: client-side validation only. Wire the submit to your
// backend or a form service (Formspree, Netlify Forms, etc.) to receive messages.
const form = document.getElementById("contact");
const status = form.querySelector(".form__status");
form.addEventListener("submit", e => {
  e.preventDefault();
  let ok = true;
  form.querySelectorAll("[required]").forEach(field => {
    const valid = field.checkValidity();
    field.setAttribute("aria-invalid", !valid);
    ok = ok && valid;
  });
  if (!ok) {
    status.textContent = "Please add your name and a valid email.";
    status.style.color = "var(--red)";
    return;
  }
  status.textContent = "Thanks! We'll be in touch within one business day.";
  status.style.color = "#2ecc71";
  form.reset();
});

document.getElementById("year").textContent = new Date().getFullYear();
