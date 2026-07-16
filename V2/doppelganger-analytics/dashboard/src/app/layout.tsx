import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ConversationProvider } from '@/contexts/ConversationContext';
import { ThemeProvider } from '@/contexts/ThemeContext';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "Doppelgänger Analytics",
  description: "Multi-platform messaging analytics — Instagram, Messenger, WhatsApp, and iMessage",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased bg-gray-50 min-h-screen font-sans`}
      >
        <ThemeProvider>
          <ConversationProvider>
            <main>
              {children}
            </main>
          </ConversationProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
