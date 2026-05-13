import { beforeEach, describe, expect, it } from 'vitest'

import { useAppStore } from '@/store/useAppStore'

describe('auth storage', () => {
  beforeEach(() => {
    useAppStore.getState().resetAll()
    localStorage.clear()
    sessionStorage.clear()
  })

  it('stores the active login in session storage so another tab can use a different user', () => {
    useAppStore.getState().setAuth('admin-token', {
      id: 'u-admin',
      username: 'admin',
      email: 'admin@test.local',
      role: 'admin',
    })

    expect(sessionStorage.getItem('deepbinder_token')).toBe('admin-token')
    expect(localStorage.getItem('deepbinder_token')).toBeNull()
  })
})
