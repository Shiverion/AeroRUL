import { useEffect, useRef, useState } from "react";

interface Option {
  value: string;
  label: string;
  color?: string;
}

interface MultiSelectProps {
  label: string;
  options: Option[];
  selected: string[];
  onChange: (values: string[]) => void;
}

export function MultiSelect({ label, options, selected, onChange }: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function toggle(value: string) {
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value]);
  }

  return (
    <div className="multiselect" ref={ref}>
      <button type="button" className="multiselect-trigger" onClick={() => setOpen((o) => !o)}>
        {label} ({selected.length}) {open ? "▴" : "▾"}
      </button>
      {open && (
        <div className="multiselect-panel">
          {options.map((opt) => {
            const isSelected = selected.includes(opt.value);
            return (
              <div
                key={opt.value}
                className={`multiselect-option ${isSelected ? "selected" : ""}`}
                onClick={() => toggle(opt.value)}
              >
                <span
                  className="swatch"
                  style={{ background: isSelected ? opt.color ?? "var(--fg)" : "transparent" }}
                />
                {opt.label}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
