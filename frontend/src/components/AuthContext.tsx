import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

interface User {
  id: number;
  email: string;
  created_at: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

// Helper to set axios auth header
function setAxiosAuthHeader(token: string | null) {
  if (token) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  } else {
    delete axios.defaults.headers.common['Authorization'];
  }
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(() => {
    const savedToken = localStorage.getItem('auth_token');
    // Set axios header immediately on initialization
    if (savedToken) {
      setAxiosAuthHeader(savedToken);
    }
    return savedToken;
  });
  const [isLoading, setIsLoading] = useState(true);
  
  // Track if we've already set user from login/register
  const hasSetUserRef = useRef(false);

  // Fetch user info on mount ONLY if token exists but user wasn't set by login/register
  useEffect(() => {
    const fetchUser = async () => {
      // Skip if no token
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Skip if user was already set by login/register
      if (hasSetUserRef.current) {
        hasSetUserRef.current = false; // Reset for next time
        setIsLoading(false);
        return;
      }

      // Ensure axios header is set
      setAxiosAuthHeader(token);

      try {
        const response = await axios.get(`${API_BASE_URL}/auth/me`);
        setUser(response.data.user);
      } catch (error) {
        // Token is invalid or expired
        console.error('Failed to fetch user:', error);
        localStorage.removeItem('auth_token');
        setToken(null);
        setUser(null);
        setAxiosAuthHeader(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUser();
  }, [token]);

  const login = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/login`, {
        email,
        password,
      });

      const { user: userData, access_token } = response.data;
      
      // Set axios header IMMEDIATELY before updating state
      setAxiosAuthHeader(access_token);
      localStorage.setItem('auth_token', access_token);
      
      // Mark that we've set the user, so useEffect doesn't re-fetch
      hasSetUserRef.current = true;
      
      setUser(userData);
      setToken(access_token);
      setIsLoading(false);
      
      return { success: true };
    } catch (error: any) {
      const errorMessage = error.response?.data?.error || 'Login failed. Please try again.';
      return { success: false, error: errorMessage };
    }
  };

  const register = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/register`, {
        email,
        password,
      });

      const { user: userData, access_token } = response.data;
      
      // Set axios header IMMEDIATELY before updating state
      setAxiosAuthHeader(access_token);
      localStorage.setItem('auth_token', access_token);
      
      // Mark that we've set the user, so useEffect doesn't re-fetch
      hasSetUserRef.current = true;
      
      setUser(userData);
      setToken(access_token);
      setIsLoading(false);
      
      return { success: true };
    } catch (error: any) {
      const errorMessage = error.response?.data?.error || 'Registration failed. Please try again.';
      return { success: false, error: errorMessage };
    }
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setToken(null);
    setUser(null);
    setAxiosAuthHeader(null);
  };

  const value: AuthContextType = {
    user,
    token,
    isLoading,
    isAuthenticated: !!user,
    login,
    register,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}
