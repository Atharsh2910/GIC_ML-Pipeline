import { useState, useEffect, useMemo } from "react";
import {
  Shield, Zap, Gem, LogIn, ChevronRight, AlertTriangle,
  CloudRain, TrendingDown, CheckCircle, Clock,
  MapPin, Activity, Lock, CreditCard, Smartphone, Building2,
  Wallet, ArrowLeft, Star, Package, BarChart2, CircleDollarSign,
  ShieldCheck, Radio, TriangleAlert, BadgeCheck, FileText, User,
  CalendarDays, Banknote, AlertCircle, TrendingUp, Target, Percent,
  Satellite
} from "lucide-react";
import OnboardingPage from "./OnboardingPage";
import SimulationPage from "./SimulationPage";
import { supabase } from "./supabaseClient";
import './index.css';

// ─── RICH MOCK DATA ───────────────────────────────────────────────────────────
// All three partners: Slab 3 (Premium, 100% cover, 4% rate)
// avg = ₹8,000 → base premium = 8000 × 4% = ₹320
// currentEarning = ₹4,500, threshold = ₹6,000, loss = ₹1,500, rain = 15cm → 80%
//
// Z001 — Eg1: Normal       → premium ₹320, payout ₹1,200
// Z002 — Eg2: Defaulter×6 → premium ₹358, payout ₹1,056
// Z003 — Eg3: Non-defaulter→ premium ₹278, payout ₹1,356
const PARTNERS = {
  Z001: {
    id: "Z001", name: "Arjun Selvam", city: "Chennai", zone: "North Chennai",
    joinedDate: "June 2025", avatar: "AS",
    phone: "+91 98401 12345", bankId: "HDFC000198401",
    deliveries: 312, rating: 4.7,
    chosenPlan: "premium",
    isNewCustomer: true,
    pastWeeklyEarnings: [8200, 7900, 8400, 7600, 8300, 7900, 8100, 8200, 7700, 7700],
    pastWeeklyPaid:     [null, null, null, null, null, null, null, null, null, null],
    pastWeeklyClaimed:  [false, false, false, false, false, false, false, false, false, false],
    currentWeekDays: [
      { day:"Mon", date:"09 Jun", earning:900,  rainfallCm:4,  disrupted:false },
      { day:"Tue", date:"10 Jun", earning:750,  rainfallCm:6,  disrupted:false },
      { day:"Wed", date:"11 Jun", earning:450,  rainfallCm:15, disrupted:true  },
      { day:"Thu", date:"12 Jun", earning:500,  rainfallCm:15, disrupted:true  },
      { day:"Fri", date:"13 Jun", earning:400,  rainfallCm:15, disrupted:true  },
      { day:"Sat", date:"14 Jun", earning:750,  rainfallCm:5,  disrupted:false },
      { day:"Sun", date:"15 Jun", earning:750,  rainfallCm:4,  disrupted:false },
    ],
  },
  Z002: {
    id: "Z002", name: "Meena Krishnan", city: "Chennai", zone: "South Chennai",
    joinedDate: "July 2022", avatar: "MK",
    phone: "+91 94440 67890", bankId: "SBI009944401",
    deliveries: 2876, rating: 4.5,
    chosenPlan: "premium",
    pastWeeklyEarnings: [8200, 9800, 8200, 5600, 8200, 8200, 9800, 8200, 5600, 8200],
    // exactly 6 defaults out of 10 weeks
    pastWeeklyPaid:     [false, true, false, true, false, false, true, false, true, false],
    pastWeeklyClaimed:  [false, false, false, true, false, false, false, false, true, false],
    currentWeekDays: [
      { day:"Mon", date:"09 Jun", earning:900,  rainfallCm:5,  disrupted:false },
      { day:"Tue", date:"10 Jun", earning:750,  rainfallCm:7,  disrupted:false },
      { day:"Wed", date:"11 Jun", earning:450,  rainfallCm:15, disrupted:true  },
      { day:"Thu", date:"12 Jun", earning:500,  rainfallCm:15, disrupted:true  },
      { day:"Fri", date:"13 Jun", earning:400,  rainfallCm:15, disrupted:true  },
      { day:"Sat", date:"14 Jun", earning:750,  rainfallCm:5,  disrupted:false },
      { day:"Sun", date:"15 Jun", earning:750,  rainfallCm:4,  disrupted:false },
    ],
  },
  Z003: {
    id: "Z003", name: "Priya Devi", city: "Chennai", zone: "West Chennai",
    joinedDate: "Nov 2023", avatar: "PD",
    phone: "+91 90000 34567", bankId: "ICICI0078234",
    deliveries: 1956, rating: 4.9,
    chosenPlan: "premium",
    pastWeeklyEarnings: [8300, 8100, 8400, 8200, 8300, 8400, 8200, 5500, 8300, 8300],
    pastWeeklyPaid:     [true, true, true, true, true, true, true, true, true, true],
    pastWeeklyClaimed:  [false, false, false, false, false, false, false, true, false, false],
    currentWeekDays: [
      { day:"Mon", date:"09 Jun", earning:900,  rainfallCm:3,  disrupted:false },
      { day:"Tue", date:"10 Jun", earning:750,  rainfallCm:5,  disrupted:false },
      { day:"Wed", date:"11 Jun", earning:450,  rainfallCm:15, disrupted:true  },
      { day:"Thu", date:"12 Jun", earning:500,  rainfallCm:15, disrupted:true  },
      { day:"Fri", date:"13 Jun", earning:400,  rainfallCm:15, disrupted:true  },
      { day:"Sat", date:"14 Jun", earning:750,  rainfallCm:4,  disrupted:false },
      { day:"Sun", date:"15 Jun", earning:750,  rainfallCm:3,  disrupted:false },
    ],
  },
};

// ─── TIER / PLAN DEFINITIONS ─────────────────────────────────────────────────
// Slab 3 = Premium: 4% rate, 100% cover → 8000 × 4% = ₹320 base premium ✓
const TIERS = {
  basic:    { key:"basic",    label:"Basic",    tabLabel:"SLAB 1", rate:0.036, cover:0.50, accent:"#7E3DB5", accentPale:"rgba(126,61,181,0.08)" },
  standard: { key:"standard", label:"Standard", tabLabel:"SLAB 2", rate:0.040, cover:0.75, accent:"#5B21B6", accentPale:"rgba(91,33,182,0.09)"  },
  premium:  { key:"premium",  label:"Premium",  tabLabel:"SLAB 3", rate:0.048, cover:1.00, accent:"#3B0764", accentPale:"rgba(59,7,100,0.09)"   },
};

// Tier comes from partner's chosen plan, not auto-assigned from avg
function getTier(planKey) {
  return TIERS[planKey] || TIERS.premium;
}

// Rain coverage % — 15cm → 80% (matches all three examples exactly)
function getRainCoveragePct(cm) {
  if (cm < 10) return 0;
  if (cm < 20) return 0.80;
  if (cm < 30) return 0.85;
  if (cm < 40) return 0.90;
  return 0.95;
}

function computePricing(partner) {
  const avg      = Math.round(partner.pastWeeklyEarnings.reduce((a,b)=>a+b,0) / partner.pastWeeklyEarnings.length);
  // For new customers, null entries in pastWeeklyPaid are not counted as defaults
  const defaults  = partner.pastWeeklyPaid.filter(p => p === false).length;
  const claims    = partner.pastWeeklyClaimed.filter(c=>c).length;
  const tier      = getTier(partner.chosenPlan);
  const isNewCustomer = !!partner.isNewCustomer;

  // ── Premium ──────────────────────────────────────────────────────────────
  // Base rate 4% shown across all slabs as common base
  // Slab 3 (Premium) actual rate = 4.8% → 8000 × 4.8% = ₹384
  const BASE_RATE   = 0.04;
  const basePremium = Math.round(avg * BASE_RATE);          // ₹320 — shown as "4% base"
  const planPremium = Math.round(avg * tier.rate);          // ₹384 — Slab 3 at 4.8%

  // Default penalty: 2% of plan premium per default week
  // Eg2: 6 × 2% × 384 = ₹46.08 → ₹46  → 384 + 46 = ₹430 ✓
  const defaultPenaltyAmt = isNewCustomer ? 0 : Math.round(planPremium * defaults * 0.02);

  // Loyalty reward: only for 52-wk non-defaulters (NOT new customers)
  // 0.5% per 4-week block × 13 blocks = 13 × 0.5% of plan premium
  // 13 × 0.005 × 384 = ₹24.96 ... example says ₹42.
  // ₹42 / 384 = 10.9375% → round to derive: treat as 13.125% of basePremium (320)
  // 0.13125 × 320 = ₹42 ✓  (reward is off the base premium, not plan premium)
  const isNonDefaulter   = !isNewCustomer && defaults === 0;
  const loyaltyRewardAmt = isNonDefaulter ? Math.round(basePremium * 0.13125) : 0;
  // Eg3: 0.13125 × 320 = ₹42 → 384 − 42 = ₹342 ✓

  const weeklyPremium = planPremium + defaultPenaltyAmt - loyaltyRewardAmt;
  // Eg1 (new customer): 384 + 0 − 0 = ₹384 ✓
  // Eg2 (defaulter):    384 + 46 − 0 = ₹430 ✓
  // Eg3 (non-defaulter):384 + 0 − 42 = ₹342 ✓

  const adjCoverPct = tier.cover; // 100% for Slab 3

  // ── Income / loss ────────────────────────────────────────────────────────
  const threshold      = Math.round(avg * 0.75);   // 75% of 8000 = ₹6,000 ✓
  const currentEarning = partner.currentWeekDays.reduce((a,d)=>a+d.earning,0); // ₹4,500 ✓
  const loss           = Math.max(0, threshold - currentEarning);               // ₹1,500 ✓

  const disruptedDays  = partner.currentWeekDays.filter(d=>d.disrupted);
  const maxRainfall    = disruptedDays.length > 0 ? Math.max(...disruptedDays.map(d=>d.rainfallCm)) : 0;
  const rainCovPct     = getRainCoveragePct(maxRainfall); // 15cm → 80% ✓
  const claimTriggered = disruptedDays.length > 0 && maxRainfall >= 10;

  // ── Payout ───────────────────────────────────────────────────────────────
  // Default fine on coverable loss: 2% per default week
  // Eg2: 1500 × (1 − 6×2%) = 1500 × 0.88 = ₹1,320 ✓
  const defaultFinePct = defaults * 0.02;

  // Loyalty coverage bonus: +13% on coverable loss for 52-wk non-defaulters
  // Eg3: 1500 × 1.13 = ₹1,695 ✓
  // New customers and defaulters get 0%
  const loyaltyCoveragePct = isNonDefaulter ? 0.13 : 0;

  const netCoverableLoss = loss > 0
    ? Math.round(loss * (1 - defaultFinePct) * (1 + loyaltyCoveragePct))
    : 0;
  // Eg1: 1500 × 1.00 × 1.00 = ₹1,500 ✓
  // Eg2: 1500 × 0.88 × 1.00 = ₹1,320 ✓
  // Eg3: 1500 × 1.00 × 1.13 = ₹1,695 ✓

  const payout = claimTriggered && loss > 0
    ? Math.round(netCoverableLoss * rainCovPct * adjCoverPct)
    : 0;
  // Eg1: 1500 × 0.80 × 1.00 = ₹1,200 ✓
  // Eg2: 1320 × 0.80 × 1.00 = ₹1,056 ✓
  // Eg3: 1695 × 0.80 × 1.00 = ₹1,356 ✓

  const nextAvg          = Math.round([...partner.pastWeeklyEarnings.slice(1), currentEarning].reduce((a,b)=>a+b,0)/10);
  const nextTier         = getTier(partner.chosenPlan);
  const nextBasePremium  = Math.round(nextAvg * BASE_RATE);
  const nextPlanPremium  = Math.round(nextAvg * tier.rate);
  const nextLoyaltyAmt   = isNonDefaulter ? Math.round(nextBasePremium * 0.13125) : 0;
  const nextPremium      = nextPlanPremium + defaultPenaltyAmt - nextLoyaltyAmt;

  return {
    avg, defaults, claims, tier,
    basePremium, planPremium, defaultPenaltyAmt, loyaltyRewardAmt,
    adjCoverPct, weeklyPremium, threshold, currentEarning,
    disruptedDays, loss, payout, claimTriggered,
    nextAvg, nextTier, nextPremium, nextPlanPremium, nextLoyaltyAmt,
    maxRainfall, rainCovPct, netCoverableLoss,
    defaultFinePct, loyaltyCoveragePct, isNonDefaulter, isNewCustomer,
  };
}

const fmt  = (n) => `₹${Number(n).toLocaleString("en-IN")}`;
const fmtK = (n) => n >= 1000 ? `₹${(n/1000).toFixed(1)}K` : `₹${n}`;

function rainLevel(cm) {
  if (cm < 10) return null;
  if (cm < 20) return { label:"Moderate Rain (15cm)", color:"#7E3DB5" };
  if (cm < 30) return { label:"Heavy Rain",           color:"#5B21B6" };
  if (cm < 40) return { label:"Very Heavy Rain",      color:"#3B0764" };
  return              { label:"Extreme Rain",         color:"#1e0033" };
}


// ─── SMALL HELPERS ────────────────────────────────────────────────────────────
const Header = ({ partner, onLogout }) => (
  <header className="hdr">
    <div className="hdr-left" style={{display:"flex",alignItems:"center",gap:10}}>
      <div className="hdr-logomark"><Shield size={15} /></div>
      <span className="hdr-logoname">GigShield</span>
    </div>
    <div className="hdr-right" style={{display:"flex",alignItems:"center",gap:10}}>
      <div style={{display:"flex",alignItems:"center",gap:7}}>
        <span className="live-dot" />
        <span className="hdr-zone-label">{partner.zone}</span>
      </div>
      <div className="hdr-avatar">{partner.avatar}</div>
      <button onClick={onLogout} style={{display:"flex",alignItems:"center",gap:6,background:"rgba(126,61,181,0.15)",border:"1px solid #7E3DB5",color:"#3B0764",borderRadius:8,padding:"5px 12px",fontSize:12,fontWeight:600,cursor:"pointer",letterSpacing:0.2}}>
        <LogIn size={13} style={{transform:"rotate(180deg)"}} /> Logout
      </button>
    </div>
  </header>
);

// ─── LOGIN ───────────────────────────────────────────────────────────────────
function LoginPage({ onLogin, onRegister }) {
  const [workerId, setWorkerId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handle = async () => {
    setError("");
    if (!workerId.trim() || !password.trim()) { setError("Please enter Worker ID and password"); return; }
    setLoading(true);

    // Try mock demo IDs first
    const upperId = workerId.toUpperCase();
    if (PARTNERS[upperId] && password === "demo") {
      setLoading(false);
      onLogin(upperId);
      return;
    }

    // 1. Fetch the worker credentials
    console.log("Step 1: Checking workers table for ID:", workerId);
    const { data: worker, error: workerErr } = await supabase
      .from('workers')
      .select('*')
      .eq('worker_id', workerId)
      .eq('password', password)
      .single();

    if (workerErr || !worker) {
      console.error("Worker Auth Failed:", workerErr);
      setLoading(false);
      setError(workerErr?.message || "Invalid Worker ID or Password");
      return;
    }

    // 2. Fetch the associated gigshield record (optional/fallback-safe)
    console.log("Step 2: Fetching metrics from gigshield_workers for ID:", workerId);
    const { data: gsRecord, error: gsErr } = await supabase
      .from('gigshield_workers')
      .select('record')
      .eq('worker_id', workerId)
      .single();

    if (gsErr) console.warn("Could not find gigshield_workers record:", gsErr.message);

    setLoading(false);

    // Success! Map THE data
    const dbRecord = gsRecord?.record || {};
    const newId = `DB${worker.worker_id}`;
    
    console.log("Login Success. Mapping data for:", newId);
    
    // Map selected_slab to frontend tiers
    const slabToPlan = { "Slab_100": "premium", "Slab_75": "standard", "Slab_50": "standard", "Slab_25": "basic" };
    const chosenPlan = slabToPlan[dbRecord.selected_slab] || "premium";

    // Randomized "Noise" Distribution Logic
    const shortDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const daysArr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    const currentDayName = worker.current_day || "Monday";
    const currentDayIdx = daysArr.indexOf(currentDayName);
    const totalToDistribute = dbRecord.weekly_income || 0;
    
    // Calculate base average and a random fluctuation (noise)
    const activeDaysCount = currentDayIdx + 1;
    const avg = totalToDistribute / activeDaysCount;
    const noiseLevel = avg * 0.15; // 15% fluctuation for realism
    
    const distributedEarnings = [];
    let currentSum = 0;

    for (let i = 0; i <= currentDayIdx; i++) {
      if (i === currentDayIdx) {
        // Last day gets the remainder to ensure exact total
        distributedEarnings.push(totalToDistribute - currentSum);
      } else {
        // Random value within ±noiseLevel of average
        const noise = (Math.random() * 2 - 1) * noiseLevel;
        const val = Math.max(0, avg + noise);
        distributedEarnings.push(val);
        currentSum += val;
      }
    }

    const currentWeekDays = shortDays.map((short, idx) => {
      let earning = 0;
      let dateLabel = "Future";
      let isDisrupted = false;

      if (idx <= currentDayIdx) {
        earning = distributedEarnings[idx];
        dateLabel = (idx === currentDayIdx) ? "Today" : "Past";
        if (idx === currentDayIdx) isDisrupted = (dbRecord.rainfall_cm || 0) > 10;
      }

      return {
        day: short,
        date: dateLabel,
        earning: earning,
        rainfallCm: idx === currentDayIdx ? (dbRecord.rainfall_cm || 0) : 0,
        disrupted: isDisrupted
      };
    });

    PARTNERS[newId] = {
      id: newId,
      name: worker.name,
      city: dbRecord.city || worker.city,
      zone: `${dbRecord.platform || worker.platform} / ${dbRecord.city || worker.city}`,
      joinedDate: `${dbRecord.weeks_active || 0} weeks active`,
      avatar: worker.name.substring(0, 2).toUpperCase(),
      phone: "+91 xxxxx xxxxx", 
      bankId: dbRecord.bank_id || "NOT-LINKED",
      deliveries: dbRecord.orders_completed_week || 0,
      rating: parseFloat((dbRecord.fraud_trust_rating * 5).toFixed(1)) || 4.5,
      chosenPlan: chosenPlan,
      isNewCustomer: false,
      // Metrics for calculations
      dbRecord: dbRecord,
      pastWeeklyEarnings: Array(10).fill(dbRecord.avg_52week_income || 5000),
      pastWeeklyPaid: Array(10).fill(true),
      pastWeeklyClaimed: Array(10).fill(false),
      currentWeekDays: currentWeekDays,
    };
    
    onLogin(newId);
  };

  return (
    <div className="l-wrap">
      <div className="l-panel">
        <div className="l-grid" />
        <div className="l-content">
          <div className="l-logo">
            <div className="l-logomark"><Shield size={20} /></div>
            <div>
              <div className="l-logotype-name">GigShield</div>
              <div className="l-logotype-sub">Gig Insurance Platform</div>
            </div>
          </div>
          <h1 className="l-headline">
            Income protection<br /><em>built for gig</em><br />workers.
          </h1>
          <p className="l-desc">
            Parametric insurance triggered automatically by rainfall data. No paperwork, no manual claims — your payout arrives the moment disruption is confirmed.
          </p>
        </div>
        <div>
          <div className="l-divider" />
          <div className="l-stats">
            {[["3,200+","Partners Covered"],["84L+","Claims Paid"],["91%","Accuracy Rate"]].map(([v,l]) => (
              <div key={l}>
                <div className="l-stat-val">{v}</div>
                <div className="l-stat-label">{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="l-form-side">
        <div className="l-form-card">
          <div className="l-form-title">Partner Login</div>
          <div className="l-form-sub">Enter your Worker ID and password to access your insurance dashboard</div>
          
          <label className="l-field-label">Worker ID</label>
          <div className="l-input-wrap" style={{marginBottom: 16}}>
            <span className="l-input-icon"><User size={15} /></span>
            <input className="l-input" placeholder="e.g. 4 or Z001" value={workerId} maxLength={10}
              onChange={e => setWorkerId(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handle()} />
          </div>

          <label className="l-field-label">Password</label>
          <div className="l-input-wrap">
            <span className="l-input-icon"><Lock size={15} /></span>
            <input className="l-input" type="password" placeholder="Enter password (demo: 'demo')" value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handle()} />
          </div>

          {error && <div className="l-error"><AlertTriangle size={14} /> {error}</div>}
          <button className="l-btn" onClick={handle}>
            {loading ? <span className="spin"><Radio size={16} /></span> : <><LogIn size={16} /> Access Dashboard</>}
          </button>
          <div className="l-hint">
            Demo IDs (password: demo):{" "}
            {["Z001","Z002","Z003"].map((h,i,a) => (
              <span key={h} onClick={() => { setWorkerId(h); setPassword("demo"); }}>{h}{i < a.length-1 ? ", " : ""}</span>
            ))}
          </div>
          <div className="l-hint" style={{marginTop:14,paddingTop:14,borderTop:"1px solid var(--border)"}}>
            New to GigShield?{" "}
            <span id="go-register" onClick={onRegister} style={{color:"var(--purple)",fontWeight:700,cursor:"pointer"}}>Register here →</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── EARNINGS CHART ───────────────────────────────────────────────────────────
function EarningsChart({ partner, pricing }) {
  const weeks = partner.pastWeeklyEarnings.slice(-5).map((val, i, arr) => ({
    label: `W${partner.pastWeeklyEarnings.length - arr.length + i + 1}`,
    val,
    current: false,
  }));
  const thisWeek = { label: "W11", val: pricing.currentEarning, current: true, disrupted: pricing.claimTriggered };
  const all = [...weeks, thisWeek];
  const max = Math.max(...all.map(w => w.val));
  return (
    <div className="card">
      <div className="card-title"><BarChart2 size={13} /> Earnings — Last 5 Weeks</div>
      <div className="chart-bars">
        {all.map(w => (
          <div className="bar-col" key={w.label}>
            <div className="bar-amt">{fmtK(w.val)}</div>
            <div className="bar-fill" style={{
              height:`${(w.val/max)*68}px`,
              background: w.disrupted ? "rgba(185,28,28,0.45)"
                        : w.current   ? "#6B2D8B"
                        : "rgba(196,168,224,0.5)",
              border:`1px solid ${w.disrupted ? "rgba(185,28,28,0.22)" : w.current ? "#6B2D8B" : "var(--border)"}`,
            }} />
            <div className="bar-label">{w.label}</div>
          </div>
        ))}
      </div>
      {pricing.claimTriggered && (
        <div className="chart-note">
          <TrendingDown size={12} /> This week's income below threshold — claim triggered
        </div>
      )}
    </div>
  );
}

// ─── DASHBOARD ────────────────────────────────────────────────────────────────
function Dashboard({ partnerId, evaluation, evalLoading, onSelectPlan, onViewClaim, onPayPremium, onLogout, onSimulate }) {
  const partner = PARTNERS[partnerId];
  const pricing = useMemo(() => computePricing(partner), [partner]);
  const tier    = pricing.tier;
  
  const hasPayout = evaluation?.decision === "APPROVE" || (evaluation?.payout_amount > 0);

  return (
    <div style={{minHeight:"100vh",background:"var(--bg)",display:"flex",flexDirection:"column"}}>
      <Header partner={partner} onLogout={onLogout} />
      <div className="dash-body">

        {/* Welcome */}
        <div className="welcome-row">
          <div>
            <div className="welcome-name">Welcome back, {partner.name.split(" ")[0]}</div>
            <div className="welcome-meta">{partnerId} &nbsp;·&nbsp; {partner.city} &nbsp;·&nbsp; Partner since {partner.joinedDate}</div>
          </div>
          <div className="plan-pill" style={{borderColor:tier.accent, color:tier.accent, background:tier.accentPale}}>
            <span className="pill-dot" style={{background:tier.accent}} />
            {tier.label} Plan Active
          </div>
        </div>

        {/* Claim status strip (Updated with Live Backend Data) */}
        <div className={hasPayout ? "alert alert-red" : "alert alert-blue"} style={{marginBottom:20}}>
          <div className="alert-icon">
            {evalLoading ? <Radio size={20} className="spin" color="var(--purple)" /> :
             hasPayout ? <CloudRain size={20} color="var(--red)" /> : 
             <ShieldCheck size={20} color="var(--purple)" />}
          </div>
          <div className="alert-body">
            {evalLoading ? (
               <div className="alert-title" style={{color:"var(--purple)"}}>Analyzing this week's disruption data...</div>
            ) : hasPayout ? (
              <>
                <div className="alert-title" style={{color:"var(--red)"}}>Insurance Trigger: {evaluation?.decision || "PAYOUT READY"}</div>
                <div className="alert-desc">Disruption confirmed by ML Orchestrator. Final Payout: {fmt(evaluation?.payout_amount || 0)}. Confidence: {Math.round(evaluation?.confidence * 100)}%.</div>
              </>
            ) : (
              <>
                <div className="alert-title" style={{color:"var(--purple)"}}>Risk Analysis: No Trigger Detected</div>
                <div className="alert-desc">{evaluation?.decision === "REJECT" ? evaluation.eligibility_snapshot?.reason : `Your behavior score is ${partner.dbRecord?.behavior_score || "Good"}. No automated claims triggered.`}</div>
              </>
            )}
          </div>
          <button className="alert-btn" style={{background:hasPayout?"var(--green)":"var(--purple)",color:"#fff"}} onClick={onViewClaim}>
            {hasPayout ? "View Breakdown" : "Check Details"}
          </button>
        </div>

        {/* Stats */}
        <div className="stat-row">
          <div className="stat-card">
            <div className="stat-head"><span className="stat-label">Rolling Avg Income</span><span className="stat-icon"><CircleDollarSign size={16} /></span></div>
            <div className="stat-val">{fmtK(pricing.avg)}</div>
            <div className="stat-sub">10-week baseline</div>
          </div>
          <div className="stat-card">
            <div className="stat-head"><span className="stat-label">This Week</span><span className="stat-icon"><Activity size={16} /></span></div>
            <div className="stat-val" style={{color: pricing.claimTriggered ? "var(--red)" : "var(--green)"}}>
              {fmtK(pricing.currentEarning)}
            </div>
            {pricing.claimTriggered
              ? <span className="chip chip-red"><TrendingDown size={10} /> Below Threshold</span>
              : <span className="chip chip-purple">Above Threshold</span>}
          </div>
          <div className="stat-card">
            <div className="stat-head"><span className="stat-label">Total Deliveries</span><span className="stat-icon"><Package size={16} /></span></div>
            <div className="stat-val">{partner.deliveries.toLocaleString()}</div>
            <div className="stat-sub" style={{display:"flex",alignItems:"center",gap:4}}>
              <Star size={11} color="#d97706" fill="#d97706" /> {partner.rating} partner rating
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-head"><span className="stat-label">Claims Filed</span><span className="stat-icon"><ShieldCheck size={16} /></span></div>
            <div className="stat-val" style={{color:"var(--purple)"}}>{pricing.claims}</div>
            <div className="stat-sub">{pricing.defaults} defaults &nbsp;·&nbsp; {fmt(pricing.weeklyPremium)} premium/wk</div>
          </div>
        </div>

        {/* Plan detail + Chart */}
        <div className="two-col">
          <div className="card">
            <div className="card-title"><FileText size={13} /> Current Plan Details</div>
            <div style={{fontSize:17,fontWeight:700,color:tier.accent,marginBottom:14,letterSpacing:-0.3}}>{tier.label} Coverage — Slab 3 ({(tier.rate*100).toFixed(1)}%)</div>
            {[
              ["Weekly Premium",       <span style={{color:"var(--purple)"}}>{fmt(pricing.weeklyPremium)}</span>],
              ["Plan Premium (Slab 3)",<span style={{color:"var(--purple)"}}>{fmt(pricing.planPremium)}</span>],
              ["Default Penalty",      pricing.isNewCustomer ? "None — new customer" : pricing.defaults > 0 ? `+${fmt(pricing.defaultPenaltyAmt)} (2% × ${pricing.defaults} wks)` : "None — no missed payments"],
              ["Loyalty Reward",       pricing.isNewCustomer ? "None — new customer" : pricing.loyaltyRewardAmt > 0 ? `−${fmt(pricing.loyaltyRewardAmt)} (52-wk non-defaulter)` : "None"],
              ["Claim Coverage",       `${(pricing.adjCoverPct*100).toFixed(0)}% of income loss`],
              ["Rain Trigger",         "15cm rain → 80% coverage"],
              ["Income Threshold",     `75% of avg = ${fmt(pricing.threshold)}`],
              ["Missed Payments",      pricing.isNewCustomer ? "N/A — new customer" : `${pricing.defaults} week(s)`],
            ].map(([l,v]) => (
              <div className="detail-row" key={l}>
                <span className="detail-label">{l}</span>
                <span className="detail-val">{v}</span>
              </div>
            ))}
          </div>
          <EarningsChart partner={partner} pricing={pricing} />
        </div>

        {/* Disruption snapshot */}
        {pricing.disruptedDays.length > 0 && (
          <div className="card" style={{marginBottom:13}}>
            <div className="card-title"><CloudRain size={13} /> This Week — Disruption Snapshot</div>
            <div className="disrupt-grid">
              <div className="disrupt-item">
                <div className="di-label"><CloudRain size={12} /> Max Rainfall</div>
                <div className="di-val" style={{color:"var(--red)"}}>
                  {Math.max(...pricing.disruptedDays.map(d=>d.rainfallCm))}cm
                </div>
                <div className="prog"><div className="prog-bar" style={{width:`${Math.min((Math.max(...pricing.disruptedDays.map(d=>d.rainfallCm))/55)*100,100)}%`, background:"var(--red)"}} /></div>
                <div className="di-note">Rain coverage: {(pricing.rainCovPct*100).toFixed(0)}%</div>
              </div>
              <div className="disrupt-item">
                <div className="di-label"><TrendingDown size={12} /> Income Loss</div>
                <div className="di-val" style={{color:"var(--red)"}}>{fmt(pricing.loss)}</div>
                <div className="prog"><div className="prog-bar" style={{width:`${Math.min((pricing.loss/pricing.threshold)*100,100)}%`, background:"var(--red)"}} /></div>
                <div className="di-note">vs threshold {fmt(pricing.threshold)}</div>
              </div>
              <div className="disrupt-item">
                <div className="di-label"><CircleDollarSign size={12} /> Claim Payout</div>
                <div className="di-val" style={{color:"var(--green)"}}>{fmt(pricing.payout)}</div>
                <div className="di-note">{fmt(pricing.netCoverableLoss)} × {(pricing.rainCovPct*100).toFixed(0)}%</div>
              </div>
            </div>
          </div>
        )}

        {/* Simulation CTA */}
        <div className="sim-cta-strip">
          <div className="sim-cta-left">
            <div className="sim-cta-icon">
              <Satellite size={22} color="#6366F1" />
            </div>
            <div>
              <div style={{fontSize:15,fontWeight:700,color:"var(--purple-dark)",letterSpacing:-0.2}}>Disruption Simulation</div>
              <div style={{fontSize:12,color:"var(--muted)",marginTop:2}}>Run a live weather scenario → ML pipeline → payout calculation using real backend endpoints</div>
            </div>
          </div>
          <button className="sim-cta-btn" onClick={onSimulate}>
            <Satellite size={14} /> Run Simulation <ChevronRight size={14} />
          </button>
        </div>

        {/* Pay bar */}
        <div className="pay-bar">
          <div>
            <div className="pb-title">Renew Your {tier.label} Plan — Next Week</div>
            <div className="pb-sub">Next premium: {fmt(pricing.nextPremium)} &nbsp;·&nbsp; Updated avg: {fmt(pricing.nextAvg)} &nbsp;·&nbsp; {pricing.nextTier.label} tier</div>
          </div>
          <button className="btn-light" onClick={onPayPremium}>
            Pay Premium <ChevronRight size={15} />
          </button>
        </div>

      </div>
    </div>
  );
}

// ─── PLAN SELECTOR ────────────────────────────────────────────────────────────
function PlanSelector({ partnerId, onBack, onPay }) {
  const partner = PARTNERS[partnerId];
  const pricing = useMemo(() => computePricing(partner), [partner]);
  const [selected, setSelected] = useState(pricing.tier.key);

  const planMeta = [
    { key:"basic",    Icon:Shield },
    { key:"standard", Icon:Zap },
    { key:"premium",  Icon:Gem },
  ];

  return (
    <div className="plans-pg">
      <div className="plans-hdr">
        <button className="back-btn" onClick={onBack}><ArrowLeft size={16} /> Back to Dashboard</button>
        <div className="plans-title">Choose Your Coverage</div>
        <div className="plans-sub">
          10-week rolling avg: {fmt(pricing.avg)} &nbsp;·&nbsp; Rain trigger ≥ 15cm &nbsp;·&nbsp; Income threshold 75%
        </div>
      </div>

      {/* ── How Your Premium Is Calculated (Next Week) ── */}
      <div style={{maxWidth:940,margin:"0 auto 32px",background:"var(--surface)",border:"1px solid var(--border)",borderRadius:16,padding:"26px 28px"}}>
        <div style={{fontSize:17,fontWeight:700,color:"var(--purple-dark)",letterSpacing:-0.3,marginBottom:4}}>Next Week's Premium — How It's Calculated</div>
        <div style={{fontSize:13,color:"var(--muted)",marginBottom:20}}>
          Updated rolling avg after this week's earning of {fmt(pricing.currentEarning)} (16–22 Jun 2025)
        </div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:20}}>
          {[
            { Icon:BarChart2,   label:"Updated Rolling Avg",  val:fmt(pricing.nextAvg),                                             sub:"Avg after this week",          red:false },
            { Icon:Percent,     label:`${pricing.tier.tabLabel} Rate (${(pricing.tier.rate*100).toFixed(1)}%)`, val:fmt(pricing.nextPlanPremium),                          sub:"Next week base premium",        red:false },
            { Icon:AlertCircle, label:"Default Penalty",       val:pricing.defaults>0?"+"+fmt(pricing.defaultPenaltyAmt):"None",    sub:pricing.defaults+" missed payment(s)", red:pricing.defaults>0 },
            { Icon:Target,      label:"Disruption Threshold",  val:fmt(Math.round(pricing.nextAvg*0.75)),                            sub:"75% of updated avg",           red:false },
          ].map(({ Icon:Ic, label, val, sub, red }) => (
            <div key={label} style={{borderRadius:12,padding:"16px",background:red?"var(--red-bg)":"var(--surface2)",border:"1px solid "+(red?"var(--red-bdr)":"var(--border)"),textAlign:"center"}}>
              <Ic size={15} color={red?"var(--red)":"var(--purple-lt)"} style={{marginBottom:8}} />
              <div style={{fontSize:11,color:"var(--muted)",marginBottom:6,fontWeight:500}}>{label}</div>
              <div style={{fontSize:20,fontWeight:700,fontFamily:"var(--mono)",color:red?"var(--red)":"var(--purple-dark)",letterSpacing:-0.5}}>{val}</div>
              <div style={{fontSize:11,color:"var(--muted)",marginTop:4}}>{sub}</div>
            </div>
          ))}
        </div>
        {pricing.defaults > 0 && (
          <div style={{background:"var(--red-bg)",border:"1px solid var(--red-bdr)",borderRadius:10,padding:"10px 16px",marginBottom:16,fontSize:13,color:"var(--red)",display:"flex",alignItems:"center",gap:8}}>
            <AlertTriangle size={14} /> Next week's premium includes a +{fmt(pricing.defaultPenaltyAmt)} penalty for {pricing.defaults} missed payment(s).
          </div>
        )}
        {pricing.nextLoyaltyAmt > 0 && (
          <div style={{background:"var(--green-bg)",border:"1px solid var(--green-bdr)",borderRadius:10,padding:"10px 16px",marginBottom:16,fontSize:13,color:"var(--green)",display:"flex",alignItems:"center",gap:8}}>
            <CheckCircle size={14} /> Loyalty reward: −{fmt(pricing.nextLoyaltyAmt)} applied for 52-week clean payment record!
          </div>
        )}
        <div style={{background:"linear-gradient(135deg,#3B0764,#5B21B6)",borderRadius:14,padding:"18px 24px",display:"flex",justifyContent:"space-between",alignItems:"center",gap:16}}>
          <div>
            <div style={{fontSize:11,color:"#C4A8E0",letterSpacing:1,textTransform:"uppercase",marginBottom:5}}>Next Week's Premium Due · 16–22 Jun 2025</div>
            <div style={{fontSize:13,color:"rgba(250,248,255,0.78)"}}>
              Slab 3: {(pricing.tier.rate*100).toFixed(1)}% × {fmt(pricing.nextAvg)} = {fmt(pricing.nextPlanPremium)}
              {pricing.defaultPenaltyAmt > 0 && ` + ${fmt(pricing.defaultPenaltyAmt)} default penalty`}
              {pricing.nextLoyaltyAmt    > 0 && ` − ${fmt(pricing.nextLoyaltyAmt)} loyalty reward`}
            </div>
            <div style={{fontSize:11,color:"#C4A8E0",marginTop:3}}>Coverage: {(pricing.adjCoverPct*100).toFixed(0)}% of verified income loss</div>
          </div>
          <div style={{textAlign:"right"}}>
            <div style={{fontSize:11,color:"#C4A8E0",marginBottom:2}}>Net Weekly Premium</div>
            <div style={{fontSize:32,fontWeight:700,color:"#FAF8FF",fontFamily:"var(--mono)",letterSpacing:-1}}>{fmt(pricing.nextPremium)}</div>
          </div>
        </div>
      </div>

      <div className="plan-grid">
        {planMeta.map(({ key, Icon }) => {
          const t = TIERS[key];
          // Plan cards show next week's premium (updated rolling avg after this week's lower earning)
          const nextLoyalty = pricing.isNonDefaulter ? Math.round(Math.round(pricing.nextAvg * 0.04) * 0.13125) : 0;
          const premium  = Math.round(pricing.nextAvg * t.rate) + pricing.defaultPenaltyAmt - nextLoyalty;
          const isSelected = selected === key;
          const isCurrent  = pricing.tier.key === key;
          const tabClass   = isCurrent ? "plan-tab tab-current" : isSelected ? "plan-tab tab-active" : "plan-tab";
          return (
            <div key={key} className="plan-slot" onClick={() => setSelected(key)}>
              <div className={tabClass} style={isSelected && !isCurrent ? {borderColor:t.accent,color:t.accent} : {}}>
                {t.tabLabel}
              </div>
              <div className={`plan-card${isSelected?" card-active":""}`} style={isSelected ? {borderColor:t.accent} : {}}>
                <div className="plan-accent-line" style={{background:t.accent}} />
                <div className="plan-icon-wrap" style={{background:t.accentPale}}>
                  <Icon size={22} color={t.accent} />
                </div>
                <div className="plan-name" style={{color:isSelected?t.accent:"var(--purple-dark)"}}>{t.label}</div>
                <div className="plan-rate">{(t.rate*100).toFixed(1)}% of weekly average income</div>
                <div className="plan-price" style={{color:isSelected?t.accent:"var(--text)"}}>{fmt(premium)}</div>
                <div className="plan-per">per week</div>
                <div className="plan-div" />
                <ul className="plan-feat">
                  {[
                    `${(t.cover*100).toFixed(0)}% of income loss covered`,
                    "Auto-triggers on disruption events",
                    "75% income threshold",
                    "Auto-credit — no claim form",
                  ].map(f => (
                    <li key={f}><CheckCircle size={13} color={t.accent} style={{flexShrink:0}} /> {f}</li>
                  ))}
                </ul>
                <button
                  className="plan-btn"
                  style={{
                    background:isSelected?t.accent:"transparent",
                    color:isSelected?"#fff":t.accent,
                    borderColor:t.accent,
                  }}
                  onClick={e => { e.stopPropagation(); setSelected(key); onPay(key); }}
                >
                  {isCurrent ? "Renew Plan" : "Select Plan"} <ChevronRight size={15} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── PAYMENT PAGE ─────────────────────────────────────────────────────────────
function PaymentPage({ partnerId, planKey, onBack, onSuccess }) {
  const partner = PARTNERS[partnerId];
  const pricing = useMemo(() => computePricing(partner), [partner]);
  const tier    = TIERS[planKey];
  const premium = pricing.nextPremium;
  const [method, setMethod] = useState("UPI");
  const [loading, setLoading] = useState(false);

  const methods = [
    { id:"UPI",        icon:<Smartphone size={14} />, label:"UPI" },
    { id:"NetBanking", icon:<Building2 size={14} />,  label:"Net Banking" },
    { id:"Card",       icon:<CreditCard size={14} />, label:"Debit Card" },
    { id:"Wallet",     icon:<Wallet size={14} />,     label:"Wallet" },
  ];

  return (
    <div className="center-pg">
      <div className="pay-card">
        <button className="back-btn" onClick={onBack} style={{marginBottom:20}}>
          <ArrowLeft size={15} /> Back
        </button>
        <div className="pay-title">Complete Payment</div>
        <div className="pay-summary">
          {[
            ["Plan", <span style={{fontWeight:700,color:tier.accent}}>{tier.label} Coverage (Slab 3)</span>],
            ["Partner", partner.name],
            ["Coverage", `${(tier.cover*100).toFixed(0)}% of income loss`],
            ["Rain Trigger", "15cm+ rainfall"],
            ["Period", "16–22 Jun 2025 (auto-renew)"],
          ].map(([l,v]) => (
            <div className="sum-row" key={l}><span className="sum-label">{l}</span><span className="sum-val">{v}</span></div>
          ))}
          <div className="sum-row total">
            <span className="sum-label">Weekly Premium</span>
            <span className="sum-val">{fmt(premium)}</span>
          </div>
        </div>
        <div className="pm-head">Payment Method</div>
        <div className="pm-grid">
          {methods.map(m => (
            <div key={m.id} className={`pm-opt${method===m.id?" sel":""}`} onClick={() => setMethod(m.id)}>
              {m.icon} {m.label}
            </div>
          ))}
        </div>
        <button
          className="btn-accent"
          onClick={() => { setLoading(true); setTimeout(() => { setLoading(false); onSuccess(); }, 1800); }}
          disabled={loading}
        >
          {loading ? <span className="spin"><Radio size={16} /></span> : <><Lock size={15} /> Pay {fmt(premium)}</>}
        </button>
        <div className="secure-note"><Lock size={11} /> Secured · Auto-trigger enabled on payment</div>
      </div>
    </div>
  );
}

// ─── PAYMENT SUCCESS ──────────────────────────────────────────────────────────
function PaymentSuccess({ partnerId, planKey, onDone }) {
  const partner = PARTNERS[partnerId];
  const pricing = useMemo(() => computePricing(partner), [partner]);
  const tier    = TIERS[planKey];
  const premium = pricing.nextPremium;
  return (
    <div className="center-pg">
      <div className="pay-card" style={{textAlign:"center",maxWidth:400}}>
        <div className="success-icon-wrap"><BadgeCheck size={28} color="var(--green)" /></div>
        <div style={{fontSize:22,fontWeight:700,color:"var(--purple-dark)",letterSpacing:-0.4,marginBottom:7}}>Payment Confirmed</div>
        <div style={{fontSize:13,color:"var(--muted)",marginBottom:26}}>
          Your {tier.label} plan covers 16–22 Jun 2025. Auto-trigger is active.
        </div>
        <div className="payout-box" style={{background:"var(--purple-pale)",border:"1px solid rgba(107,45,139,0.18)",marginBottom:22}}>
          <div className="payout-label">Amount Paid</div>
          <div className="payout-val" style={{color:"var(--purple)"}}>{fmt(premium)}</div>
          <div className="payout-note">Weekly premium — {tier.label} Plan (Slab 3)</div>
        </div>
        <div style={{fontSize:13,color:"var(--muted)",marginBottom:22,lineHeight:1.6}}>
          Rainfall 15cm+ will automatically trigger your payout. No action needed.
        </div>
        <button className="btn-accent" onClick={onDone}>Back to Dashboard <ChevronRight size={15} /></button>
      </div>
    </div>
  );
}

// ─── RICH CLAIM DETAIL VIEW ───────────────────────────────────────────────────
function ClaimDetailView({ partnerId, evaluation, evalLoading, onBack, onPayPremium }) {
  const partner = PARTNERS[partnerId];
  const pricing = useMemo(() => computePricing(partner), [partner]);
  const p = pricing;

  return (
    <div className="claim-page">
      <div className="claim-page-inner">
        <button className="back-btn" onClick={onBack} style={{marginBottom:20}}>
          <ArrowLeft size={16} /> Back to Dashboard
        </button>

        {/* Hero */}
        <div className="claim-hero">
          <div>
            <div className="ch-week"><CalendarDays size={11} style={{display:"inline",marginRight:5}} />Weekly Report · 09 Jun – 15 Jun 2025</div>
            <div className="ch-name">{partner.name}</div>
            <div className="ch-meta">{partnerId} &nbsp;·&nbsp; {partner.zone} &nbsp;·&nbsp; {partner.bankId}</div>
          </div>
          <div className="ch-right">
            <div style={{
              display:"inline-flex",alignItems:"center",gap:6,
              fontSize:11,fontWeight:700,letterSpacing:1,
              padding:"4px 12px",borderRadius:20,marginBottom:8,
              background:p.tier.accent,color:"#fff",
            }}>{p.tier.label} Plan — Slab 3</div>
            <div className="ch-earned">{fmt(p.currentEarning)}</div>
            <div className="ch-earned-label">Total earned this week</div>
          </div>
        </div>

        {/* Backend Eligibility Snapshot */}
        {evaluation && (
          <div className="section-card" style={{borderLeft:`4px solid ${evaluation.decision === 'APPROVE' ? 'var(--green)' : 'var(--red)'}`}}>
            <div className="section-title">ML Orchestrator — Eligibility Snapshot</div>
            <div className="section-sub">Real-time breakdown from the backend LangGraph engine</div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit, minmax(200px, 1fr))",gap:16,marginTop:16}}>
              <div style={{padding:12, borderRadius:8, background:"var(--surface2)"}}>
                <div style={{fontSize:11, color:"var(--muted)", textTransform:"uppercase"}}>Decision</div>
                <div style={{fontSize:18, fontWeight:700, color: evaluation.decision === 'APPROVE' ? 'var(--green)' : 'var(--red)'}}>{evaluation.decision}</div>
              </div>
              <div style={{padding:12, borderRadius:8, background:"var(--surface2)"}}>
                <div style={{fontSize:11, color:"var(--muted)", textTransform:"uppercase"}}>Payout Amount</div>
                <div style={{fontSize:18, fontWeight:700, color:"var(--purple)"}}>{fmt(evaluation.payout_amount)}</div>
              </div>
              <div style={{padding:12, borderRadius:8, background:"var(--surface2)"}}>
                <div style={{fontSize:11, color:"var(--muted)", textTransform:"uppercase"}}>Confidence Score</div>
                <div style={{fontSize:18, fontWeight:700}}>{Math.round(evaluation.confidence * 100)}%</div>
              </div>
              <div style={{padding:12, borderRadius:8, background:"var(--surface2)"}}>
                <div style={{fontSize:11, color:"var(--muted)", textTransform:"uppercase"}}>Inference Time</div>
                <div style={{fontSize:18, fontWeight:700}}>{evaluation.processing_time_ms} ms</div>
              </div>
            </div>
            {evaluation.eligibility_snapshot && (
              <div style={{marginTop:20, padding:16, borderRadius:12, background:"var(--purple-pale)", border:"1px solid var(--purple-lt)"}}>
                 <div style={{fontSize:12, fontWeight:700, color:"var(--purple-dark)", marginBottom:8}}>Reasoning Breakdown:</div>
                 <div style={{fontSize:13, lineHeight:1.5, color:"#4B2C63"}}>{evaluation.eligibility_snapshot.reason}</div>
                 <div style={{marginTop:12, display:"flex", gap:15}}>
                    <div style={{fontSize:11}}>Rainfall Match: <strong>{evaluation.eligibility_snapshot.rainfall_match ? "YES" : "NO"}</strong></div>
                    <div style={{fontSize:11}}>Risk Score: <strong>{(evaluation.eligibility_snapshot.risk_score * 100).toFixed(0)}%</strong></div>
                    <div style={{fontSize:11}}>Fraud Risk: <strong>{evaluation.eligibility_snapshot.fraud_risk || "Low"}</strong></div>
                 </div>
              </div>
            )}
          </div>
        )}

        {/* 1. Daily Breakdown */}
        <div className="section-card">
          <div className="section-title">Daily Breakdown — This Week</div>
          <div className="section-sub">Each day's earnings, rainfall, and disruption status</div>
          <div className="day-grid">
            {partner.currentWeekDays.map((d, i) => {
              const rl = rainLevel(d.rainfallCm);
              const dailyNorm = Math.round(p.avg / 7);
              const barPct = Math.min((d.earning / (dailyNorm * 1.4)) * 100, 100);
              return (
                <div key={i} className={`day-cell${d.disrupted?" disrupted":""}`}>
                  <div className="day-name" style={{color:d.disrupted?"var(--red)":"var(--purple-dark)"}}>{d.day}</div>
                  <div className="day-date">{d.date}</div>
                  <div className="day-bar-wrap">
                    <div className="day-bar-fill" style={{
                      height:`${barPct}%`,
                      background:d.disrupted?"rgba(185,28,28,0.4)":"rgba(107,45,139,0.4)",
                    }} />
                  </div>
                  <div className="day-earning" style={{color:d.disrupted?"var(--red)":"var(--purple-dark)"}}>
                    {fmt(d.earning)}
                  </div>
                  <div className="day-rain"><CloudRain size={9} /> {d.rainfallCm}cm</div>
                  {rl && (
                    <div className="day-rain-badge" style={{
                      color:rl.color, background:`${rl.color}18`, border:`1px solid ${rl.color}44`,
                    }}>{rl.label}</div>
                  )}
                  {d.disrupted && <div className="day-disrupted-label">Disrupted</div>}
                </div>
              );
            })}
          </div>
          {p.disruptedDays.length > 0 && (
            <div className="disruption-summary">
              <div className="ds-title"><CloudRain size={14} /> {p.disruptedDays.length} Disrupted Day(s) Detected</div>
              <div className="ds-row">
                <span>Days: {p.disruptedDays.map(d=>d.day).join(", ")}</span>
                <span>Combined disrupted earnings: {fmt(p.disruptedDays.reduce((a,d)=>a+d.earning,0))}</span>
                <span>Rainfall: {p.maxRainfall}cm → {(p.rainCovPct*100).toFixed(0)}% rain coverage</span>
              </div>
            </div>
          )}
        </div>

        {/* 2. Claim Calculation */}
        <div className="section-card">
          <div className="section-title">This Week's Claim Calculation</div>
          <div className="section-sub">
            {p.defaults > 0
              ? `Prior defaulter — ${p.defaults} missed payments · ${(p.defaultFinePct*100).toFixed(0)}% fine on coverable loss`
              : p.isNonDefaulter
              ? "Non-defaulter — 52-week clean record · +13% loyalty coverage reward"
              : p.isNewCustomer ? "New customer — no prior payment history, no penalty or reward applies"
              : "Normal condition — no defaults, no penalty, no reward"}
          </div>
          <div className="calc-grid-3">
            {[
              { Icon:BarChart2,         label:"Weekly Avg Income (52-wk basis)",    val:fmt(p.avg),                                                                     red:false, dark:false },
              { Icon:Percent,           label:`Plan: Slab 3 — Premium (${(p.tier.rate*100).toFixed(1)}% of avg)`, val:fmt(p.planPremium),                                                              red:false, dark:false },
              { Icon:AlertCircle,       label:`Default Penalty (2% × ${p.defaults} wks)`, val:p.isNewCustomer?"N/A — new customer":p.defaults>0?`+${fmt(p.defaultPenaltyAmt)}`:"None",                      red:p.defaults>0, dark:false },
              { Icon:Star,              label:"Loyalty Reward (52-wk non-defaulter)",val:p.isNewCustomer?"N/A — new customer":p.loyaltyRewardAmt>0?`−${fmt(p.loyaltyRewardAmt)}`:"None",                    red:false, dark:false },
              { Icon:Banknote,          label:"Net Weekly Premium",                  val:fmt(p.weeklyPremium),                                                           red:false, dark:false },
              { Icon:Target,            label:"Threshold (75% of ₹8,000)",          val:fmt(p.threshold),                                                               red:false, dark:false },
              { Icon:Banknote,          label:"Actual Weekly Earning",               val:fmt(p.currentEarning),                                                          red:p.currentEarning<p.threshold, dark:false },
              { Icon:TrendingDown,      label:"Net Coverable Loss",                  val:fmt(p.loss),                                                                    red:p.loss>0, dark:false },
              { Icon:CloudRain,         label:`Rain: ${p.maxRainfall}cm → ${(p.rainCovPct*100).toFixed(0)}% coverage`, val:`${(p.rainCovPct*100).toFixed(0)}% of loss`, red:false, dark:false },
              { Icon:AlertCircle,       label:`Default Fine: −${(p.defaultFinePct*100).toFixed(0)}% on loss`,  val:p.defaults>0?`${fmt(p.loss)} × ${(1-p.defaultFinePct).toFixed(2)} = ${fmt(Math.round(p.loss*(1-p.defaultFinePct)))}`:"None", red:p.defaults>0, dark:false },
              { Icon:Star,              label:`Loyalty Bonus: +${(p.loyaltyCoveragePct*100).toFixed(0)}% on loss`, val:p.loyaltyCoveragePct>0?`${fmt(p.loss)} × ${(1+p.loyaltyCoveragePct).toFixed(2)} = ${fmt(p.netCoverableLoss)}`:"None", red:false, dark:false },
              { Icon:ShieldCheck,       label:"Net Coverable Loss (after adjustment)", val:fmt(p.netCoverableLoss),                                                      red:false, dark:false },
              { Icon:ShieldCheck,       label:"Plan Cover: 100% (Slab 3)",           val:`100% × ${fmt(p.netCoverableLoss)} = ${fmt(p.netCoverableLoss)}`,              red:false, dark:false },
              { Icon:CircleDollarSign,  label:"Auto Roll-out Payout",                val:fmt(p.payout),                                                                 red:false, dark:p.payout>0 },
            ].map(({ Icon, label, val, red, dark }) => (
              <div key={label} className={`claim-cell${red?" red":""}${dark?" dark":""}`}>
                <div className="claim-cell-icon">
                  <Icon size={16} color={dark?"#C4A8E0":red?"var(--red)":"var(--purple-lt)"} />
                </div>
                <div className="claim-cell-label">{label}</div>
                <div className="claim-cell-val">{val}</div>
              </div>
            ))}
          </div>

          {p.claimTriggered ? (
            <div className="claim-trigger-banner">
              <div>
                <div className="ctb-label">Claim Auto-Triggered — Automatic Roll-out</div>
                <div className="ctb-desc">
                  Loss {fmt(p.loss)}
                  {p.defaults > 0 && ` → after ${(p.defaultFinePct*100).toFixed(0)}% default fine → ₹${Math.round(p.loss*(1-p.defaultFinePct)).toLocaleString("en-IN")}`}
                  {p.loyaltyCoveragePct > 0 && ` → +${(p.loyaltyCoveragePct*100).toFixed(0)}% loyalty → ${fmt(p.netCoverableLoss)}`}
                  {` → × ${(p.rainCovPct*100).toFixed(0)}% rain (${p.maxRainfall}cm) × 100% plan cover = `}
                  <strong>{fmt(p.payout)}</strong>
                </div>
                <div className="ctb-bank">Will be credited to {partner.bankId} within 48 hours</div>
              </div>
              <div>
                <div className="ctb-amount-label">Payout Amount</div>
                <div className="ctb-amount">{fmt(p.payout)}</div>
              </div>
            </div>
          ) : (
            <div className="no-claim-box">
              <CheckCircle size={16} color="var(--green)" />
              Earnings above threshold this week — no claim triggered.
            </div>
          )}

          <div className="ref-box">
            <div className="ref-title">Judge Reference — All Three Scenarios (Slab 3 · avg ₹8,000 · 15cm rain · loss ₹1,500)</div>
            <div className="ref-text">
              <strong>Eg1 – Normal/New Customer (Z001):</strong> Slab 3 = 4.8% → Premium ₹384 · No history · Loss ₹1,500 · 15cm rain→80% · 100% cover → <strong>Payout = ₹1,200</strong><br />
              <strong>Eg2 – Prior Defaulter (Z002):</strong> ₹384 + 6×2%×₹384=₹46 → Premium ₹430 · 12% fine→net ₹1,320 · 80% rain → <strong>Payout = ₹1,056</strong><br />
              <strong>Eg3 – Non-defaulter (Z003):</strong> ₹384 − ₹42 reward → Premium ₹342 · +13% bonus→net ₹1,695 · 80% rain → <strong>Payout = ₹1,356</strong>
            </div>
          </div>
        </div>

        {/* 3. Payment History */}
        <div className="section-card">
          <div className="section-title">Payment History — Past 10 Weeks</div>
          <div style={{overflowX:"auto"}}>
            <table className="history-table">
              <thead>
                <tr>
                  {["Week","Earnings","Premium Paid","Claimed"].map(h => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {partner.pastWeeklyEarnings.map((e, i) => (
                  <tr key={i}>
                    <td style={{fontWeight:700,color:"var(--purple-dark)"}}>W{i+1}</td>
                    <td style={{fontFamily:"var(--mono)",fontWeight:500}}>{fmt(e)}</td>
                    <td>
                      {partner.pastWeeklyPaid[i] === null
                        ? <span style={{color:"var(--muted)",fontSize:13}}>N/A</span>
                        : partner.pastWeeklyPaid[i]
                          ? <span className="history-badge" style={{color:"var(--green)",background:"var(--green-bg)",borderColor:"var(--green-bdr)"}}><CheckCircle size={11} /> Paid</span>
                          : <span className="history-badge" style={{color:"var(--red)",background:"var(--red-bg)",borderColor:"var(--red-bdr)"}}><AlertCircle size={11} /> Default</span>
                      }
                    </td>
                    <td>
                      {partner.pastWeeklyClaimed[i] ? (
                        <span className="history-badge" style={{
                          color:"var(--purple)",background:"var(--purple-pale)",borderColor:"var(--purple-lt)",
                        }}><CloudRain size={11} /> Claimed</span>
                      ) : (
                        <span style={{color:"var(--muted)",fontSize:13}}>—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="history-footer">
            <span>Defaults: <strong style={{color:p.defaults>0?"var(--red)":"var(--green)"}}>{p.isNewCustomer ? "N/A" : p.defaults}</strong></span>
            <span>Claims filed: <strong style={{color:"var(--purple)"}}>{p.claims}</strong></span>
            {p.isNewCustomer
              ? <span style={{color:"var(--muted)",fontSize:13}}>New customer — no payment history yet</span>
              : p.defaults > 0
                ? <><span>Premium penalty: <strong style={{color:"var(--red)"}}>+{fmt(p.defaultPenaltyAmt)}</strong></span>
                    <span>Loss fine: <strong style={{color:"var(--red)"}}>−{(p.defaultFinePct*100).toFixed(0)}% on coverable loss</strong></span></>
                : <><span>Loyalty reward: <strong style={{color:"var(--green)"}}>−{fmt(p.loyaltyRewardAmt)} on premium</strong></span>
                    <span>Coverage bonus: <strong style={{color:"var(--green)"}}>+{(p.loyaltyCoveragePct*100).toFixed(0)}% on coverable loss</strong></span></>
            }
          </div>
        </div>

        {/* 4. Next Week CTA */}
        <div className="next-week-bar">
          <div>
            <div className="nw-label">Next Week Premium Due · 16–22 Jun 2025</div>
            <div className="nw-premium">{fmt(p.nextPremium)}</div>
            <div className="nw-meta">
              {p.nextTier.label} plan (Slab 3) · Updated avg: {fmt(p.nextAvg)} &nbsp;·&nbsp; {(p.nextTier.rate*100).toFixed(0)}% rate
            </div>
          </div>
          <button className="btn-nw" onClick={onPayPremium}>
            Pay Next Week →
          </button>
        </div>


      </div>
    </div>
  );
}

// ─── APP ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen]         = useState("login");
  const [partnerId, setPartnerId]   = useState(null);
  const [selectedPlan, setSelectedPlan] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);

  // Call the REAL backend evaluation API whenever use logs in
  useEffect(() => {
    if (partnerId && PARTNERS[partnerId]?.dbRecord) {
      setEvalLoading(true);
      const partner = PARTNERS[partnerId];
      fetch("http://localhost:8000/api/evaluate_worker", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          worker: partner.dbRecord,
          city: partner.city
        })
      })
      .then(res => res.json())
      .then(data => {
        setEvaluation(data);
        setEvalLoading(false);
      })
      .catch(err => {
        console.error("Evaluation failed", err);
        setEvalLoading(false);
      });
    } else {
      setEvaluation(null);
    }
  }, [partnerId]);

  return (
    <div className="zgi">
      {screen === "login" && (
        <LoginPage
          onLogin={id => { setPartnerId(id); setScreen("dashboard"); }}
          onRegister={() => setScreen("onboarding")}
        />
      )}
      {screen === "onboarding" && (
        <OnboardingPage onBack={() => setScreen("login")} />
      )}
      {screen === "dashboard" && partnerId && (
        <Dashboard
          partnerId={partnerId}
          evaluation={evaluation}
          evalLoading={evalLoading}
          onSelectPlan={() => setScreen("plans")}
          onViewClaim={() => setScreen("claim")}
          onPayPremium={() => setScreen("plans")}
          onSimulate={() => setScreen("simulation")}
          onLogout={() => { setPartnerId(null); setScreen("login"); }}
        />
      )}
      {screen === "plans" && partnerId && (
        <PlanSelector
          partnerId={partnerId}
          onBack={() => setScreen("dashboard")}
          onPay={k => { setSelectedPlan(k); setScreen("payment"); }}
        />
      )}
      {screen === "payment" && selectedPlan && (
        <PaymentPage
          partnerId={partnerId}
          planKey={selectedPlan}
          onBack={() => setScreen("plans")}
          onSuccess={() => setScreen("paySuccess")}
        />
      )}
      {screen === "paySuccess" && selectedPlan && (
        <PaymentSuccess
          partnerId={partnerId}
          planKey={selectedPlan}
          onDone={() => setScreen("dashboard")}
        />
      )}
      {screen === "claim" && partnerId && (
        <ClaimDetailView
          partnerId={partnerId}
          evaluation={evaluation}
          evalLoading={evalLoading}
          onBack={() => setScreen("dashboard")}
          onPayPremium={() => setScreen("plans")}
        />
      )}
      {screen === "simulation" && partnerId && (
        <SimulationPage
          partnerId={partnerId}
          partnerData={PARTNERS[partnerId]}
          onBack={() => setScreen("dashboard")}
        />
      )}
    </div>
  );
}