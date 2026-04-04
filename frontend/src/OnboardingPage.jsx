import { useState } from "react";
import {
  Shield, User, Lock, MapPin, Smartphone, Hash, Building2,
  ChevronRight, ArrowLeft, CheckCircle, AlertTriangle, Radio,
  BadgeCheck, Zap,
} from "lucide-react";
import { supabase } from "./supabaseClient";

// ── Platform options ────────────────────────────────────────────────────────
const PLATFORMS = ["Blinkit", "Zepto", "Swiggy", "Zomato", "Dunzo", "BigBasket", "Other"];

// ── Step indicator ──────────────────────────────────────────────────────────
function StepDots({ step }) {
  return (
    <div className="ob-stepper">
      {[1, 2].map((s) => (
        <div key={s} className="ob-step-item">
          <div
            className={`ob-step-dot ${step === s ? "active" : step > s ? "done" : ""}`}
          >
            {step > s ? <CheckCircle size={12} /> : s}
          </div>
          {s < 2 && <div className={`ob-step-line ${step > s ? "done" : ""}`} />}
        </div>
      ))}
    </div>
  );
}

// ── Field wrapper ────────────────────────────────────────────────────────────
function Field({ label, icon: Icon, error, children }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <label className="l-field-label">{label}</label>
      <div className="l-input-wrap">
        <span className="l-input-icon">
          <Icon size={15} />
        </span>
        {children}
      </div>
      {error && (
        <div className="l-error" style={{ marginTop: 6 }}>
          <AlertTriangle size={13} /> {error}
        </div>
      )}
    </div>
  );
}

// ── Step 1 — Personal Info ───────────────────────────────────────────────────
function Step1({ form, setForm, errors }) {
  return (
    <>
      <Field label="Full Name" icon={User} error={errors.name}>
        <input
          id="ob-name"
          className="l-input"
          placeholder="e.g. Arjun Selvam"
          value={form.name}
          onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
        />
      </Field>
      <Field label="Password" icon={Lock} error={errors.password}>
        <input
          id="ob-password"
          type="password"
          className="l-input"
          placeholder="Create a password"
          value={form.password}
          onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
        />
      </Field>
    </>
  );
}

// ── Step 2 — Work Details ────────────────────────────────────────────────────
function Step2({ form, setForm, errors }) {
  return (
    <>
      <Field label="City" icon={MapPin} error={errors.city}>
        <input
          id="ob-city"
          className="l-input"
          placeholder="e.g. Coimbatore"
          value={form.city}
          onChange={(e) => setForm((f) => ({ ...f, city: e.target.value }))}
        />
      </Field>

      <Field label="Platform" icon={Smartphone} error={errors.platform}>
        <select
          id="ob-platform"
          className="l-input ob-select"
          value={form.platform}
          onChange={(e) => setForm((f) => ({ ...f, platform: e.target.value }))}
        >
          <option value="">Select platform</option>
          {PLATFORMS.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </Field>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Field label="Outlet ID" icon={Building2} error={errors.outlet_id}>
          <input
            id="ob-outlet"
            type="number"
            className="l-input"
            placeholder="e.g. 2636"
            value={form.outlet_id}
            onChange={(e) => setForm((f) => ({ ...f, outlet_id: e.target.value }))}
          />
        </Field>
        <Field label="Worker ID" icon={Hash} error={errors.worker_id}>
          <input
            id="ob-worker"
            type="number"
            className="l-input"
            placeholder="e.g. 4"
            value={form.worker_id}
            onChange={(e) => setForm((f) => ({ ...f, worker_id: e.target.value }))}
          />
        </Field>
      </div>
    </>
  );
}

// ── Success screen ───────────────────────────────────────────────────────────
function SuccessScreen({ form, onBack }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div className="ob-success-icon">
        <BadgeCheck size={32} color="var(--green)" />
      </div>
      <div className="ob-success-title">You're all set!</div>
      <div style={{ fontSize: 14, color: "var(--muted)", marginBottom: 28, lineHeight: 1.6 }}>
        Welcome to GigShield, <strong style={{ color: "var(--purple-dark)" }}>{form.name}</strong>!
        Your account is registered. Login now to access your dashboard.
      </div>

      {/* Summary card */}
      <div className="ob-summary-card">
        {[
          ["City",      form.city],
          ["Platform",  form.platform],
          ["Outlet ID", form.outlet_id],
          ["Worker ID", form.worker_id],
        ].map(([label, val]) => (
          <div key={label} className="sum-row">
            <span className="sum-label">{label}</span>
            <span className="sum-val">{val}</span>
          </div>
        ))}
      </div>

      <button className="l-btn" onClick={onBack} style={{ marginTop: 6 }}>
        <Shield size={15} /> Back to Login
      </button>
    </div>
  );
}

// ── Main Onboarding Page ─────────────────────────────────────────────────────
export default function OnboardingPage({ onBack }) {
  const [step, setStep]     = useState(1);   // 1, 2, or "done"
  const [loading, setLoading] = useState(false);
  const [errors, setErrors]   = useState({});
  const [globalError, setGlobalError] = useState("");

  const [form, setForm] = useState({
    name: "", password: "", city: "", platform: "", outlet_id: "", worker_id: "",
  });

  // ── Validation ─────────────────────────────────────────────────────────────
  function validate(stepNum) {
    const errs = {};
    if (stepNum === 1) {
      if (!form.name.trim())                       errs.name     = "Name is required";
      if (form.password.length < 6)                errs.password = "Minimum 6 characters";
    }
    if (stepNum === 2) {
      if (!form.city.trim())                       errs.city      = "City is required";
      if (!form.platform)                          errs.platform  = "Select a platform";
      if (!form.outlet_id || isNaN(form.outlet_id)) errs.outlet_id = "Valid outlet ID required";
      if (!form.worker_id || isNaN(form.worker_id)) errs.worker_id = "Valid worker ID required";
    }
    return errs;
  }

  // ── Step 1 → 2 ────────────────────────────────────────────────────────────
  function handleNext() {
    const errs = validate(1);
    if (Object.keys(errs).length) { setErrors(errs); return; }
    setErrors({});
    setStep(2);
  }

  // ── Step 2 → submit ───────────────────────────────────────────────────────
  async function handleSubmit() {
    const errs = validate(2);
    if (Object.keys(errs).length) { setErrors(errs); return; }
    setErrors({});
    setGlobalError("");
    setLoading(true);

    const payload = {
      name:      form.name.trim(),
      password:  form.password,
      city:      form.city.trim(),
      platform:  form.platform,
      outlet_id: parseInt(form.outlet_id, 10),
      worker_id: parseInt(form.worker_id, 10),
    };

    const { error } = await supabase.from("workers").insert([payload]);

    setLoading(false);
    if (error) {
      if (error.code === "23505") {
        setGlobalError("This Worker ID is already registered. Try a different one.");
      } else {
        setGlobalError(error.message || "Something went wrong. Please try again.");
      }
      return;
    }
    setStep("done");
  }

  return (
    <div className="l-wrap">
      {/* ── Left gradient panel ── */}
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
            Register once,<br /><em>protected</em><br />always.
          </h1>
          <p className="l-desc">
            Join thousands of gig workers who get automatic income protection against rainfall disruptions — no paperwork, no manual claims.
          </p>
        </div>
        <div>
          <div className="l-divider" />
          <div className="l-stats">
            {[["3,200+", "Partners Covered"], ["84L+", "Claims Paid"], ["91%", "Accuracy Rate"]].map(([v, l]) => (
              <div key={l}>
                <div className="l-stat-val">{v}</div>
                <div className="l-stat-label">{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Right form panel ── */}
      <div className="l-form-side">
        <div className="l-form-card" style={{ maxWidth: 400 }}>

          {/* Back to login */}
          {step !== "done" && (
            <button
              className="back-btn"
              onClick={onBack}
              style={{ marginBottom: 20, padding: 0 }}
            >
              <ArrowLeft size={14} /> Back to Login
            </button>
          )}

          {step !== "done" ? (
            <>
              <div className="l-form-title">
                {step === 1 ? "Create Account" : "Work Details"}
              </div>
              <div className="l-form-sub">
                {step === 1
                  ? "Set up your personal credentials to get started"
                  : "Tell us where and how you work"}
              </div>

              {/* Step dots */}
              <StepDots step={step} />

              {/* Global error */}
              {globalError && (
                <div className="l-error" style={{ marginBottom: 14, padding: "10px 14px", background: "var(--red-bg)", border: "1px solid var(--red-bdr)", borderRadius: 9 }}>
                  <AlertTriangle size={14} /> {globalError}
                </div>
              )}

              {/* Form steps */}
              {step === 1 && <Step1 form={form} setForm={setForm} errors={errors} />}
              {step === 2 && <Step2 form={form} setForm={setForm} errors={errors} />}

              {/* Actions */}
              <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
                {step === 2 && (
                  <button
                    className="back-btn"
                    onClick={() => { setStep(1); setErrors({}); }}
                    style={{ marginBottom: 0, border: "1.5px solid var(--border)", borderRadius: 10, padding: "13px 16px" }}
                  >
                    <ArrowLeft size={14} />
                  </button>
                )}
                <button
                  id="ob-next-btn"
                  className="l-btn"
                  style={{ flex: 1, marginTop: 0 }}
                  onClick={step === 1 ? handleNext : handleSubmit}
                  disabled={loading}
                >
                  {loading
                    ? <span className="spin"><Radio size={16} /></span>
                    : step === 1
                      ? <><Zap size={15} /> Continue <ChevronRight size={15} /></>
                      : <><CheckCircle size={15} /> Register</>}
                </button>
              </div>

              <div className="l-hint" style={{ marginTop: 18 }}>
                Already have an account?{" "}
                <span onClick={onBack}>Sign in</span>
              </div>
            </>
          ) : (
            <SuccessScreen form={form} onBack={onBack} />
          )}
        </div>
      </div>
    </div>
  );
}
