import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "@/components/Layout";
import Dashboard from "@/pages/Dashboard";
import PatientAnalysis from "@/pages/PatientAnalysis";
import ModelExplorer from "@/pages/ModelExplorer";
import DatasetInspector from "@/pages/DatasetInspector";
import Training from "@/pages/Training";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/"          element={<Dashboard />} />
        <Route path="/patient"   element={<PatientAnalysis />} />
        <Route path="/model"     element={<ModelExplorer />} />
        <Route path="/datasets"  element={<DatasetInspector />} />
        <Route path="/training"  element={<Training />} />
      </Route>
    </Routes>
  );
}