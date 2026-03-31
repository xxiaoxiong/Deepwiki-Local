'use client';

import React, { useState, useEffect } from 'react';

export interface ModelPreset {
  name: string;
  provider: string;
  base_url?: string;
  api_key?: string;
}

export async function fetchModelPresets(): Promise<ModelPreset[]> {
  try {
    const res = await fetch('/api/models/presets');
    if (!res.ok) return [];
    const data = await res.json();
    return data.presets || [];
  } catch {
    return [];
  }
}

interface Provider {
  id: string;
  name: string;
}

interface ModelManagementModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const EMPTY_FORM = { name: '', provider: '', base_url: '', api_key: '' };

export default function ModelManagementModal({ isOpen, onClose }: ModelManagementModalProps) {
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    fetchModelPresets().then(setPresets);
    fetch('/api/models/config')
      .then(r => r.json())
      .then(data => {
        const list: Provider[] = data.providers || [];
        setProviders(list);
        if (!form.provider && list.length > 0) {
          setForm(f => ({ ...f, provider: list[0].id }));
        }
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const handleAdd = async () => {
    if (!form.name.trim() || !form.provider) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch('/api/models/presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: form.name.trim(),
          provider: form.provider,
          base_url: form.base_url.trim() || null,
          api_key: form.api_key.trim() || null,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPresets(data.presets || []);
      setForm(f => ({ ...f, name: '', base_url: '', api_key: '' }));
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (index: number) => {
    try {
      const res = await fetch(`/api/models/presets/${index}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPresets(data.presets || []);
    } catch (e) {
      setError(String(e));
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4 bg-black/50">
        <div className="relative bg-[var(--card-bg)] rounded-lg shadow-xl w-full max-w-lg">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-color)]">
            <h3 className="text-lg font-medium text-[var(--accent-primary)]">模型预设管理</h3>
            <button
              onClick={onClose}
              className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Body */}
          <div className="p-6">
            <p className="text-xs text-[var(--muted)] mb-4">
              预设保存在服务器端，所有访问用户均可在配置页和对话模式中一键回填。
            </p>

            {/* Preset list */}
            {presets.length > 0 ? (
              <div className="mb-4 space-y-2 max-h-56 overflow-y-auto pr-1">
                {presets.map((preset, i) => (
                  <div
                    key={i}
                    className="flex items-start justify-between px-3 py-2.5 rounded-md bg-[var(--background)]/50 border border-[var(--border-color)]/50"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm text-[var(--foreground)] font-medium">{preset.name}</span>
                        <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--accent-primary)]/15 text-[var(--accent-primary)]">{preset.provider}</span>
                      </div>
                      {preset.base_url && (
                        <div className="text-xs text-[var(--muted)] truncate mt-0.5" title={preset.base_url}>
                          URL: {preset.base_url}
                        </div>
                      )}
                      {preset.api_key && (
                        <div className="text-xs text-[var(--muted)] mt-0.5">
                          Key: {'•'.repeat(Math.min(preset.api_key.length, 8))}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => handleDelete(i)}
                      className="ml-3 flex-shrink-0 text-xs text-red-400 hover:text-red-500 px-2 py-1 rounded hover:bg-red-50/10 transition-colors"
                    >
                      删除
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mb-4 text-xs text-[var(--muted)] text-center py-6 border border-dashed border-[var(--border-color)]/40 rounded-md">
                暂无预设模型，请在下方添加
              </div>
            )}

            {/* Add form */}
            <div className="border-t border-[var(--border-color)]/30 pt-4 space-y-2">
              <div className="text-xs font-medium text-[var(--foreground)] mb-2">添加新模型</div>

              <select
                value={form.provider}
                onChange={e => setForm(f => ({ ...f, provider: e.target.value }))}
                className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
              >
                {providers.length === 0 && <option value="">加载中...</option>}
                {providers.map(p => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>

              <input
                type="text"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                placeholder="模型名称，如 QWQ3-32b"
                className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
              />

              <input
                type="text"
                value={form.base_url}
                onChange={e => setForm(f => ({ ...f, base_url: e.target.value }))}
                placeholder="API Base URL（可选），如 http://10.0.0.1:8000/v1"
                className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
              />

              <div className="flex gap-2">
                <input
                  type="password"
                  value={form.api_key}
                  onChange={e => setForm(f => ({ ...f, api_key: e.target.value }))}
                  placeholder="API Key（可选）"
                  className="input-japanese block flex-1 px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
                />
                <button
                  onClick={handleAdd}
                  disabled={!form.name.trim() || !form.provider || saving}
                  className="px-3 py-1.5 text-xs rounded-md bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/80 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap transition-colors"
                >
                  {saving ? '...' : '添加'}
                </button>
              </div>

              {error && <div className="text-xs text-red-400 mt-1">{error}</div>}
            </div>
          </div>

          {/* Footer */}
          <div className="flex justify-end px-6 py-4 border-t border-[var(--border-color)]">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium rounded-md bg-[var(--accent-primary)]/90 text-white hover:bg-[var(--accent-primary)] transition-colors"
            >
              完成
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
