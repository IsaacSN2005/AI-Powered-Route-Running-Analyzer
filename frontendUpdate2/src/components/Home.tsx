import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Activity, LogIn, ChevronRight, Shield, Zap, BarChart3, AlertCircle, Loader2 } from 'lucide-react';
import type { User } from '../types';

interface HomeProps {
  onLoginSuccess: (userData: User) => void;
}

export default function Home({ onLoginSuccess }: HomeProps) {
  const [error, setError] = useState<string | null>(null);
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  const handleLogin = async () => {
    setError(null);
    setIsLoggingIn(true);
    
    // Simulate a brief delay for the login process
    setTimeout(() => {
      const mockUser = {
        uid: 'mock-user-123',
        displayName: 'Elite Athlete',
        email: 'athlete@example.com',
        photoURL: 'https://picsum.photos/seed/athlete/200/200'
      };
      onLoginSuccess(mockUser);
      setIsLoggingIn(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-primary/30">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-black/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-primary rounded-lg p-1.5 flex items-center justify-center">
              <Activity className="text-white w-6 h-6" />
            </div>
            <span className="text-xl font-bold tracking-tight">RouteIQ</span>
          </div>
          <button 
            onClick={handleLogin}
            disabled={isLoggingIn}
            className="flex items-center gap-2 bg-white text-black px-5 py-2 rounded-full font-bold text-sm hover:bg-white/90 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoggingIn ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <LogIn className="w-4 h-4" />
            )}
            Sign In
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold tracking-widest uppercase mb-6">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                </span>
                Next-Generation Wide Receiver Analytics
              </div>
              <h1 className="text-7xl lg:text-8xl font-black tracking-tighter leading-[0.9] mb-8">
                MASTER THE <br />
                <span className="text-primary">SCIENCE</span> OF <br />
                THE ROUTE.
              </h1>
              <p className="text-xl text-slate-400 max-w-lg mb-10 leading-relaxed">
                Professional-grade route running analysis powered by computer vision. 
                Upload side-view footage to track sharpness, deceleration, and re-acceleration with elite precision.
              </p>
              <div className="mb-10 rounded-2xl border border-white/10 bg-white/5 px-5 py-4 max-w-xl">
                <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400 mb-3">Currently Supported Routes</p>
                <div className="flex flex-wrap gap-2">
                  {['Comeback', 'Curl', 'Out', 'Slant', 'Dig'].map((route) => (
                    <span
                      key={route}
                      className="rounded-full border border-primary/25 bg-primary/10 px-3 py-1 text-xs font-bold text-primary"
                    >
                      {route}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex flex-wrap gap-4 items-center">
                <button 
                  onClick={handleLogin}
                  disabled={isLoggingIn}
                  className="group flex items-center gap-3 bg-primary text-white px-8 py-4 rounded-full font-bold text-lg hover:bg-primary/90 transition-all shadow-2xl shadow-primary/20 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoggingIn ? "Signing in..." : "Get Started"}
                  {!isLoggingIn && <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />}
                </button>

                <AnimatePresence>
                  {error && (
                    <motion.div 
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                      className="flex items-center gap-2 text-red-400 text-sm font-medium bg-red-400/10 border border-red-400/20 px-4 py-2 rounded-full"
                    >
                      <AlertCircle className="w-4 h-4" />
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.2 }}
              className="relative aspect-square lg:aspect-video rounded-3xl overflow-hidden border border-white/10 shadow-2xl"
            >
              <img 
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuDV6fN6kXyWzcmfShRvE9AveOpNPfWuGX9DxYCGy9un7AkQe1yKdSqZookn7JIvcz_qL3CJsMTgDw5rZ9fSU5gpyqeSU5iEBtsoXLvGYwq-3od2jap39eI26MDxxPDuO78QE2eh68gBG0LIAwNPvK300zv6yA20LEfnVVmNA1MQkNEP5zEMQoBGh87uCJI5pa8VAyvQ-H9qZaCDo2uMDnz1od3nEAKt3pUFB2TomiNEmFxpS_Yrz24rz_Qk_y5YxcKXUEG2F5FGenM" 
                alt="Wide Receiver Route Analysis"
                className="w-full h-full object-cover opacity-60"
                referrerPolicy="no-referrer"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent"></div>
              
              {/* Floating UI Elements */}
              <div className="absolute bottom-8 left-8 right-8 grid grid-cols-3 gap-4">
                {[
                  { label: 'SHARPNESS', value: '94%', icon: Zap },
                  { label: 'DECELERATION', value: '-4.2m/s²', icon: BarChart3 },
                  { label: 'SCORE', value: '92', icon: Shield },
                ].map((stat, i) => (
                  <div key={i} className="bg-black/60 backdrop-blur-md border border-white/10 p-4 rounded-2xl">
                    <stat.icon className="w-4 h-4 text-primary mb-2" />
                    <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{stat.label}</div>
                    <div className="text-xl font-black">{stat.value}</div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </main>

    </div>
  );
}
