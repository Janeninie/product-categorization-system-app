import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Product Categorization System",
  description: "ML-powered product categorization for beverages and snacks",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
