import React from 'react';
import { Home, Image, FileVideo, BarChart3, Info, HelpCircle, Shield, LogIn, LogOut, User } from 'lucide-react';
import { Button } from './ui/button';
import { useAuth } from './AuthContext';
import { ThemeToggle } from './ThemeToggle';

interface LayoutProps {
  children: React.ReactNode;
  currentPage: string;
  onNavigate: (page: string) => void;
}

export function Layout({ children, currentPage, onNavigate }: LayoutProps) {
  const { user, isAuthenticated, logout } = useAuth();

  const navItems = [
    { id: 'home', label: 'Dashboard', icon: Home },
    { id: 'analyze', label: 'Analysis', icon: Shield },
    { id: 'results', label: 'Results & Reports', icon: BarChart3 },
    { id: 'about', label: 'About', icon: Info },
    { id: 'help', label: 'Help', icon: HelpCircle },
  ];

  const handleLogout = () => {
    logout();
    onNavigate('home');
  };

  return (
    <div className="min-h-screen bg-background relative">
      {/* Header */}
      <header className="bg-card/80 backdrop-blur-md border-b border-border sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center shadow-lg shadow-primary/30">
                <Shield className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-foreground">Deepfake Detection System</h1>
                <p className="text-primary text-sm">AI-Driven Spatiotemporal Analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <ThemeToggle />
              {isAuthenticated ? (
                <>
                  <div className="flex items-center gap-2 text-right">
                    <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center">
                      <User className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <p className="text-sm text-foreground font-medium">{user?.email}</p>
                      <p className="text-xs text-muted-foreground">Logged in</p>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleLogout}
                    className="flex items-center gap-2"
                  >
                    <LogOut className="w-4 h-4" />
                    Logout
                  </Button>
                </>
              ) : (
                <>
                  <div className="text-right">
                    <p className="text-sm text-foreground">Academic Research Project</p>
                    <p className="text-xs text-muted-foreground">Version 1.0</p>
                  </div>
                  <Button
                    variant="default"
                    size="sm"
                    onClick={() => onNavigate('login')}
                    className="flex items-center gap-2 bg-primary hover:bg-primary/90"
                  >
                    <LogIn className="w-4 h-4" />
                    Sign In
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="flex relative z-10">
        {/* Sidebar */}
        <aside className="w-64 bg-sidebar backdrop-blur-md border-r border-sidebar-border min-h-[calc(100vh-73px)] p-4">
          <nav className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    currentPage === item.id
                      ? 'bg-accent text-accent-foreground border border-border'
                      : 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-accent-foreground'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </button>
              );
            })}
            
            {/* Auth navigation items */}
            {!isAuthenticated && (
              <div className="pt-4 mt-4 border-t border-border">
                <button
                  onClick={() => onNavigate('login')}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    currentPage === 'login'
                      ? 'bg-accent text-accent-foreground border border-border'
                      : 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-accent-foreground'
                  }`}
                >
                  <LogIn className="w-5 h-5" />
                  <span>Sign In</span>
                </button>
              </div>
            )}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8">
          {children}
        </main>
      </div>
    </div>
  );
}
