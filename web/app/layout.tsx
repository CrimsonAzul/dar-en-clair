import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dar en Clair",
  description:
    "Questions sur la location, la vente, la fiscalité immobilière et la copropriété au Maroc, avec les articles de loi cités.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
