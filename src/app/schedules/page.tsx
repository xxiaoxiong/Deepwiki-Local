'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { FaHome, FaPlus, FaTrash, FaEdit, FaPlay, FaSync, FaCheckCircle, FaExclamationTriangle, FaClock } from 'react-icons/fa';

interface Schedule {
  id: string;
  name: string;
  repo_url: string;
  repo_type: string;
  access_token: string;
  interval_hours: number;
  cron_expr: string;
  enabled: boolean;
  excluded_dirs: string;
  excluded_files: string;
  last_run: string;
  last_status: string;
  next_run: string;
  created_at: string;
}

const REPO_TYPES = ['gitlab', 'github', 'gitea', 'gitee', 'bitbucket'];

const emptyForm = (): Partial<Schedule> => ({
  name: '',
  repo_url: '',
  repo_type: 'gitlab',
  access_token: '',
  interval_hours: 24,
  cron_expr: '',
  enabled: true,
  excluded_dirs: '',
  excluded_files: '',
});

export default function SchedulesPage() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<Partial<Schedule>>(emptyForm());
  const [submitting, setSubmitting] = useState(false);
  const [triggeringId, setTriggeringId] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const fetchSchedules = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/schedules');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSchedules(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load schedules');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSchedules();
  }, [fetchSchedules]);

  const showSuccess = (msg: string) => {
    setSuccessMsg(msg);
    setTimeout(() => setSuccessMsg(null), 3000);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const url = editingId ? `/api/schedules/${editingId}` : '/api/schedules';
      const method = editingId ? 'PUT' : 'POST';
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }
      showSuccess(editingId ? '任务已更新' : '任务已创建');
      setShowForm(false);
      setEditingId(null);
      setForm(emptyForm());
      fetchSchedules();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save');
    } finally {
      setSubmitting(false);
    }
  };

  const handleEdit = (s: Schedule) => {
    setEditingId(s.id);
    setForm({
      name: s.name,
      repo_url: s.repo_url,
      repo_type: s.repo_type,
      access_token: s.access_token,
      interval_hours: s.interval_hours,
      cron_expr: s.cron_expr,
      enabled: s.enabled,
      excluded_dirs: s.excluded_dirs,
      excluded_files: s.excluded_files,
    });
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm('确定要删除该定时任务吗？')) return;
    try {
      const res = await fetch(`/api/schedules/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      showSuccess('任务已删除');
      fetchSchedules();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete');
    }
  };

  const handleTrigger = async (id: string) => {
    setTriggeringId(id);
    try {
      const res = await fetch(`/api/schedules/${id}/trigger`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      showSuccess('任务已手动触发，正在后台运行...');
      setTimeout(fetchSchedules, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to trigger');
    } finally {
      setTriggeringId(null);
    }
  };

  const handleToggleEnabled = async (s: Schedule) => {
    try {
      const res = await fetch(`/api/schedules/${s.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !s.enabled }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      fetchSchedules();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to update');
    }
  };

  const cancelForm = () => {
    setShowForm(false);
    setEditingId(null);
    setForm(emptyForm());
  };

  const formatTime = (iso: string) => {
    if (!iso) return '—';
    try {
      return new Date(iso).toLocaleString('zh-CN');
    } catch {
      return iso;
    }
  };

  const statusBadge = (status: string) => {
    if (!status) return null;
    if (status === 'success')
      return <span className="inline-flex items-center gap-1 text-green-600 text-xs"><FaCheckCircle />成功</span>;
    if (status === 'running')
      return <span className="inline-flex items-center gap-1 text-blue-500 text-xs"><FaSync className="animate-spin" />运行中</span>;
    if (status.startsWith('error'))
      return <span className="inline-flex items-center gap-1 text-red-500 text-xs" title={status}><FaExclamationTriangle />失败</span>;
    return <span className="text-xs text-gray-500">{status}</span>;
  };

  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)] p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Link href="/" className="text-[var(--accent-primary)] hover:opacity-80 transition-opacity">
              <FaHome size={20} />
            </Link>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <FaClock className="text-[var(--accent-primary)]" />
              定时拉取任务管理
            </h1>
          </div>
          <div className="flex gap-2">
            <button
              onClick={fetchSchedules}
              className="flex items-center gap-2 px-3 py-2 rounded-md border border-[var(--border-color)] hover:bg-[var(--background)]/80 text-sm transition-colors"
            >
              <FaSync className={loading ? 'animate-spin' : ''} size={13} />
              刷新
            </button>
            <button
              onClick={() => { cancelForm(); setShowForm(true); }}
              className="flex items-center gap-2 px-4 py-2 rounded-md bg-[var(--accent-primary)] text-white text-sm hover:opacity-90 transition-opacity"
            >
              <FaPlus size={13} />
              新建任务
            </button>
          </div>
        </div>

        {/* Messages */}
        {successMsg && (
          <div className="mb-4 px-4 py-2 rounded-md bg-green-100 text-green-800 border border-green-300 text-sm">
            {successMsg}
          </div>
        )}
        {error && (
          <div className="mb-4 px-4 py-2 rounded-md bg-red-100 text-red-800 border border-red-300 text-sm flex justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="ml-4 font-bold">×</button>
          </div>
        )}

        {/* Create / Edit Form */}
        {showForm && (
          <div className="mb-6 p-5 rounded-lg border border-[var(--border-color)] bg-[var(--background)]/50">
            <h2 className="text-lg font-semibold mb-4">{editingId ? '编辑任务' : '新建定时拉取任务'}</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">任务名称 *</label>
                  <input
                    required
                    value={form.name || ''}
                    onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                    placeholder="例如：DeepWiki 每日同步"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">仓库类型</label>
                  <select
                    value={form.repo_type || 'gitlab'}
                    onChange={e => setForm(f => ({ ...f, repo_type: e.target.value }))}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-[var(--background)] text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                  >
                    {REPO_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">仓库 URL *</label>
                <input
                  required
                  value={form.repo_url || ''}
                  onChange={e => setForm(f => ({ ...f, repo_url: e.target.value }))}
                  className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                  placeholder="http://172.16.x.x:8081/group/project.git"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">访问令牌（私有仓库）</label>
                <input
                  type="password"
                  value={form.access_token || ''}
                  onChange={e => setForm(f => ({ ...f, access_token: e.target.value }))}
                  className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                  placeholder="留空则为公开仓库"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">拉取间隔（小时）</label>
                  <input
                    type="number"
                    min={1}
                    value={form.interval_hours ?? 24}
                    onChange={e => setForm(f => ({ ...f, interval_hours: Number(e.target.value) }))}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                  />
                  <p className="text-xs text-[var(--muted)] mt-1">Cron 表达式优先于此设置</p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Cron 表达式（可选）</label>
                  <input
                    value={form.cron_expr || ''}
                    onChange={e => setForm(f => ({ ...f, cron_expr: e.target.value }))}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                    placeholder="0 2 * * *（每天凌晨 2 点）"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">排除目录（逗号分隔）</label>
                  <input
                    value={form.excluded_dirs || ''}
                    onChange={e => setForm(f => ({ ...f, excluded_dirs: e.target.value }))}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                    placeholder="node_modules, .git, dist"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">排除文件（逗号分隔）</label>
                  <input
                    value={form.excluded_files || ''}
                    onChange={e => setForm(f => ({ ...f, excluded_files: e.target.value }))}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border-color)] bg-transparent text-sm focus:outline-none focus:border-[var(--accent-primary)]"
                    placeholder="*.lock, *.log"
                  />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="enabled"
                  checked={form.enabled ?? true}
                  onChange={e => setForm(f => ({ ...f, enabled: e.target.checked }))}
                  className="w-4 h-4"
                />
                <label htmlFor="enabled" className="text-sm">启用该任务</label>
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-5 py-2 rounded-md bg-[var(--accent-primary)] text-white text-sm hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                  {submitting ? '保存中...' : (editingId ? '保存修改' : '创建任务')}
                </button>
                <button
                  type="button"
                  onClick={cancelForm}
                  className="px-4 py-2 rounded-md border border-[var(--border-color)] text-sm hover:bg-[var(--background)]/80 transition-colors"
                >
                  取消
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Schedule List */}
        {loading ? (
          <div className="text-center py-12 text-[var(--muted)]">加载中...</div>
        ) : schedules.length === 0 ? (
          <div className="text-center py-12 text-[var(--muted)] border border-dashed border-[var(--border-color)] rounded-lg">
            <FaClock size={32} className="mx-auto mb-3 opacity-40" />
            <p>暂无定时任务</p>
            <p className="text-sm mt-1">点击「新建任务」添加自动拉取分析的仓库</p>
          </div>
        ) : (
          <div className="space-y-3">
            {schedules.map(s => (
              <div
                key={s.id}
                className={`p-4 rounded-lg border transition-colors ${
                  s.enabled
                    ? 'border-[var(--border-color)] bg-[var(--background)]/40'
                    : 'border-[var(--border-color)]/50 bg-[var(--background)]/20 opacity-60'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold truncate">{s.name}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full border border-[var(--border-color)] text-[var(--muted)]">
                        {s.repo_type}
                      </span>
                      {!s.enabled && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-200 text-gray-600">已禁用</span>
                      )}
                    </div>
                    <p className="text-sm text-[var(--muted)] truncate mb-2">{s.repo_url}</p>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--muted)]">
                      <span className="flex items-center gap-1">
                        <FaClock size={10} />
                        {s.cron_expr ? `Cron: ${s.cron_expr}` : `每 ${s.interval_hours}h`}
                      </span>
                      {s.last_run && (
                        <span>上次: {formatTime(s.last_run)} {statusBadge(s.last_status)}</span>
                      )}
                      {s.next_run && (
                        <span>下次: {formatTime(s.next_run)}</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => handleTrigger(s.id)}
                      disabled={triggeringId === s.id}
                      title="立即运行"
                      className="p-2 rounded-md border border-[var(--border-color)] hover:bg-green-50 hover:border-green-400 hover:text-green-600 transition-colors disabled:opacity-50"
                    >
                      {triggeringId === s.id
                        ? <FaSync size={13} className="animate-spin" />
                        : <FaPlay size={13} />}
                    </button>
                    <button
                      onClick={() => handleToggleEnabled(s)}
                      title={s.enabled ? '禁用' : '启用'}
                      className="p-2 rounded-md border border-[var(--border-color)] hover:bg-[var(--background)]/80 transition-colors"
                    >
                      {s.enabled ? '⏸' : '▶'}
                    </button>
                    <button
                      onClick={() => handleEdit(s)}
                      title="编辑"
                      className="p-2 rounded-md border border-[var(--border-color)] hover:bg-blue-50 hover:border-blue-400 hover:text-blue-600 transition-colors"
                    >
                      <FaEdit size={13} />
                    </button>
                    <button
                      onClick={() => handleDelete(s.id)}
                      title="删除"
                      className="p-2 rounded-md border border-[var(--border-color)] hover:bg-red-50 hover:border-red-400 hover:text-red-500 transition-colors"
                    >
                      <FaTrash size={13} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
