import React from 'react';
import jsPDF from 'jspdf';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Shield, AlertTriangle, CheckCircle, Download, ArrowLeft } from 'lucide-react';
import { Progress } from './ui/progress';

interface AnalysisResult {
  fileName: string;
  type: 'image' | 'video';
  isDeepfake: boolean;
  confidence: number;
  probability: number;
  threshold?: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
  };
  detectedAnomalies: string[];
  ganFingerprint: boolean;
  ganScore?: number;
  maxFrameScore?: number;
  spatiotemporalScore?: number;
  frameScores?: number[];
  attentionScores?: number[];
  ganFamily?: string;
  ganFamilyConfidence?: number;
  familyProbs?: Record<string, number>;
  scoreSource?: string;
  filePreview?: string;
}

interface AnalysisPageProps {
  result: AnalysisResult | null;
  onBack: () => void;
}

export function AnalysisPage({ result, onBack }: AnalysisPageProps) {
  if (!result) {
    return (
      <div className="max-w-4xl mx-auto">
        <Card className="shadow-md bg-card border-border">
          <CardContent className="pt-6 text-center py-12">
            <Shield className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">No analysis results available</p>
            <p className="text-sm text-muted-foreground mt-2">Upload a file to begin analysis</p>
            <Button onClick={onBack} className="mt-6 bg-primary hover:bg-primary/90">
              Back to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const formatScoreSource = (source?: string) => {
    if (!source || source === 'unknown') return '';
    return source.replace(/\+/g, ' + ').replace(/[-_]/g, ' ').replace(/\s+/g, ' ').trim();
  };

  const downloadReport = () => {
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 48;
    let y = 36;

    const colors = {
      primary: [46, 120, 255],
      accent: [99, 102, 241],
      success: [16, 185, 129],
      danger: [220, 38, 38],
      text: [30, 41, 59],
      muted: [100, 116, 139],
      border: [226, 232, 240],
      soft: [248, 250, 252],
    };

    const addBadge = (text: string, color: number[]) => {
      const padX = 10;
      const textWidth = doc.getTextWidth(text);
      const width = textWidth + padX * 2;
      doc.setFillColor(...color);
      doc.roundedRect(margin, y - 10, width, 22, 6, 6, 'F');
      doc.setFontSize(11);
      doc.setFont(undefined, 'bold');
      doc.setTextColor(255, 255, 255);
      doc.text(text, margin + padX, y + 5);
      doc.setTextColor(...colors.text);
    };

    const addSection = (title: string) => {
      y += 10;
      doc.setFillColor(...colors.soft);
      doc.roundedRect(margin, y - 8, pageWidth - margin * 2, 30, 6, 6, 'F');
      doc.setFontSize(13);
      doc.setFont(undefined, 'bold');
      doc.setTextColor(...colors.text);
      doc.text(title, margin + 8, y + 10);
      y += 30;
    };

    const addLabelValueAt = (
      x: number,
      yPos: number,
      label: string,
      value: string,
      boldValue = false
    ) => {
      doc.setFontSize(11);
      doc.setTextColor(...colors.muted);
      doc.text(label, x, yPos);
      doc.setTextColor(...colors.text);
      doc.setFont(undefined, boldValue ? 'bold' : 'normal');
      doc.text(value, x + 120, yPos);
      doc.setFont(undefined, 'normal');
      return yPos + 16;
    };

    const addTableRow = (x: number, width: number, yPos: number, cells: string[], bold = false) => {
      const cellWidth = width / cells.length;
      doc.setFontSize(10.5);
      doc.setFont(undefined, bold ? 'bold' : 'normal');
      doc.setTextColor(...colors.text);
      cells.forEach((c, i) => {
        doc.text(c, x + 8 + cellWidth * i, yPos);
      });
      return yPos + 16;
    };

    const addProgressInline = (label: string, value: number, x: number, yPos: number, maxWidth: number, color: number[]) => {
      const clamped = Math.max(0, Math.min(100, value));
      doc.setFontSize(10.5);
      doc.setTextColor(...colors.text);
      doc.text(label, x, yPos + 4);
      const barY = yPos + 8;
      doc.setDrawColor(...colors.border);
      doc.roundedRect(x, barY, maxWidth, 9, 3, 3);
      doc.setFillColor(...color);
      doc.roundedRect(x, barY, (maxWidth * clamped) / 100, 9, 3, 3, 'F');
      doc.setTextColor(...colors.text);
      doc.text(`${clamped.toFixed(1)}%`, x + maxWidth + 10, yPos + 12);
      return yPos + 24;
    };

    const drawBarChart = (scores: number[], labels: string[], title: string) => {
      if (scores.length === 0) return;
      const chartHeight = 80;
      const chartWidth = pageWidth - margin * 2;
      doc.setFontSize(11);
      doc.setFont(undefined, 'bold');
      doc.text(title, margin, y);
      y += 6;
      const maxVal = Math.max(...scores, 1);
      const barWidth = chartWidth / (scores.length * 1.8);
      scores.forEach((s, idx) => {
        const barH = (s / maxVal) * chartHeight;
        const x = margin + idx * barWidth * 1.8;
        const barY = y + chartHeight - barH;
        doc.setFillColor(...colors.accent);
        doc.rect(x, barY, barWidth, barH, 'F');
        doc.setFontSize(9);
        doc.setTextColor(...colors.muted);
        doc.text(labels[idx], x, y + chartHeight + 10, { angle: 0 });
      });
      y += chartHeight + 24;
    };

    const drawSparkline = (values: number[], title: string) => {
      if (!values || values.length < 2) return;
      const width = pageWidth - margin * 2;
      const height = 50;
      const startY = y + 6;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min || 1;
      const points = values.map((v, i) => {
        const x = margin + (width * i) / (values.length - 1);
        const yv = startY + height - ((v - min) / range) * height;
        return [x, yv];
      });
      doc.setFontSize(11);
      doc.setFont(undefined, 'bold');
      doc.text(title, margin, y);
      doc.setDrawColor(...colors.accent);
      doc.setLineWidth(1.2);
      for (let i = 0; i < points.length - 1; i++) {
        const [x1, y1] = points[i];
        const [x2, y2] = points[i + 1];
        doc.line(x1, y1, x2, y2);
      }
      doc.setFillColor(...colors.accent);
      const [lastX, lastY] = points[points.length - 1];
      doc.circle(lastX, lastY, 2, 'F');
      y += height + 24;
    };

    // Header band
    doc.setFillColor(...colors.primary);
    doc.rect(0, 0, pageWidth, 82, 'F');
    // Logo mark
    doc.setFillColor(255, 255, 255);
    doc.circle(margin + 12, 32, 12, 'F');
    doc.setFillColor(...colors.accent);
    doc.circle(margin + 12, 32, 6, 'F');
    doc.setFontSize(20);
    doc.setFont(undefined, 'bold');
    doc.setTextColor(255, 255, 255);
    doc.text('Deepfake Detection Report', margin + 36, 34);
    doc.setFontSize(10.5);
    doc.setFont(undefined, 'normal');
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin + 36, 52);
    doc.text(`Model: ${result.modelName || 'Unknown'} | Threshold: ${result.threshold ?? 50}%`, margin + 36, 66);
    y = 102;
    const outcomeText = result.isDeepfake ? 'Deepfake Detected' : 'Authentic Media';
    const outcomeColor = result.isDeepfake ? colors.danger : colors.success;
    addBadge(outcomeText, outcomeColor);
    y += 28;

    // Narrative summary
    const ganSignalText = result.ganScore !== undefined ? `${result.ganScore}%` : '--';
    const spatioText = result.spatiotemporalScore !== undefined ? `${result.spatiotemporalScore}%` : 'N/A';
    const summary = `System classified this as ${outcomeText.toLowerCase()} with ${result.confidence}% confidence. GAN signal ${ganSignalText}${result.maxFrameScore ? ` (peak ${result.maxFrameScore}%)` : ''}; temporal score ${spatioText}.`;
    doc.setFontSize(11.5);
    doc.setTextColor(...colors.text);
    const wrapped = doc.splitTextToSize(summary, pageWidth - margin * 2);
    doc.text(wrapped, margin, y);
    y += wrapped.length * 14 + 6;

    // File + Technical (two columns)
    addSection('Key Details');
    const colWidth = (pageWidth - margin * 2 - 12) / 2;
    const col1X = margin;
    const col2X = margin + colWidth + 12;
    let colY1 = y;
    let colY2 = y;
    colY1 = addLabelValueAt(col1X, colY1, 'File', result.fileName, true);
    colY1 = addLabelValueAt(col1X, colY1, 'Type', result.type.toUpperCase());
    colY1 = addLabelValueAt(col1X, colY1, 'Outcome', outcomeText, true);
    colY1 = addLabelValueAt(col1X, colY1, 'Confidence', `${result.confidence}%`, true);
    colY2 = addLabelValueAt(col2X, colY2, 'Model', result.modelName || 'Unknown');
    colY2 = addLabelValueAt(col2X, colY2, 'Threshold', `${result.threshold ?? 50}%`);
    if (result.scoreSource && formatScoreSource(result.scoreSource)) {
      colY2 = addLabelValueAt(col2X, colY2, 'Score Source', formatScoreSource(result.scoreSource) || '--');
    }
    colY2 = addLabelValueAt(col2X, colY2, 'GAN Family', result.ganFamily 
      ? (result.ganFamilyConfidence !== undefined && result.ganFamilyConfidence !== null 
        ? `${result.ganFamily} (${result.ganFamilyConfidence.toFixed(1)}%)` 
        : result.ganFamily)
      : 'Not available');
    y = Math.max(colY1, colY2) + 6;

    // Performance metrics bars only (clean, consistent)
    addSection('Performance Metrics');
    const tableWidth = pageWidth - margin * 2;
    doc.setFillColor(...colors.soft);
    doc.roundedRect(margin, y - 10, tableWidth, 20, 8, 8, 'F');
    doc.setFontSize(11.5);
    doc.setFont(undefined, 'bold');
    doc.setTextColor(...colors.text);
    doc.text('Summary', margin + 10, y + 5);
    doc.setFontSize(10.5);
    doc.setFont(undefined, 'normal');
    doc.setTextColor(...colors.muted);
    doc.text(
      `Acc ${result.metrics.accuracy}% | Prec ${result.metrics.precision}% | Recall ${result.metrics.recall}% | Fake Prob ${result.probability}% | Conf ${result.confidence}%`,
      margin + 90,
      y + 5
    );
    y += 28;
    const barMaxWidth = tableWidth - 160;
    y = addProgressInline('Accuracy', result.metrics.accuracy, margin, y, barMaxWidth, colors.primary);
    y = addProgressInline('Precision', result.metrics.precision, margin, y, barMaxWidth, colors.accent);
    y = addProgressInline('Recall', result.metrics.recall, margin, y, barMaxWidth, colors.success);
    y = addProgressInline('Fake Probability', result.probability, margin, y, barMaxWidth, colors.danger);
    y = addProgressInline('Confidence', result.confidence, margin, y, barMaxWidth, colors.primary);

    // Technical analysis (two columns)
    addSection('Technical Analysis');
    let techY1 = y;
    let techY2 = y;
    techY1 = addLabelValueAt(col1X, techY1, 'GAN Fingerprint', result.ganFingerprint ? 'Detected' : 'Not Found', result.ganFingerprint);
    const ganSignal = result.ganScore !== undefined ? `${result.ganScore}%` : '--';
    const ganPeak = result.maxFrameScore !== undefined ? ` (peak ${result.maxFrameScore}%)` : '';
    techY1 = addLabelValueAt(col1X, techY1, 'GAN Signal', `${ganSignal}${ganPeak}`);
    if (result.spatiotemporalScore !== undefined) {
      techY1 = addLabelValueAt(col1X, techY1, 'Spatiotemporal Score', `${result.spatiotemporalScore}%`);
    }
    techY2 = addLabelValueAt(col2X, techY2, 'Media Type', result.type.toUpperCase());
    if (result.scoreSource && formatScoreSource(result.scoreSource)) {
      techY2 = addLabelValueAt(col2X, techY2, 'Pipeline', formatScoreSource(result.scoreSource) || '--');
    }
    y = Math.max(techY1, techY2) + 10;

    // Family Probabilities breakdown (if available)
    if (result.familyProbs && Object.keys(result.familyProbs).length > 0) {
      doc.setFontSize(11);
      doc.setFont(undefined, 'bold');
      doc.text('GAN Family Probabilities:', margin, y);
      y += 16;
      const sortedFamilies = Object.entries(result.familyProbs)
        .sort(([, a], [, b]) => (b as number) - (a as number));
      sortedFamilies.forEach(([family, prob]) => {
        doc.setFontSize(10);
        doc.setFont(undefined, 'normal');
        doc.setTextColor(...colors.text);
        doc.text(`• ${family}: ${(prob as number).toFixed(1)}%`, margin + 10, y);
        y += 14;
      });
      y += 6;
    }

    // Charts
    const frameScores = Array.isArray(result.frameScores) ? [...result.frameScores] : [];
    const topFrames = frameScores
      .map((s, idx) => ({ score: s * 100, idx: idx + 1 }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    if (topFrames.length > 0) {
      drawBarChart(
        topFrames.map((t) => t.score),
        topFrames.map((t) => `F${t.idx}`),
        'Top Suspicious Frames'
      );
    }
    if (result.attentionScores && result.attentionScores.length > 1) {
      const att = result.attentionScores.map((a) => a * 100);
      drawSparkline(att, 'Temporal Attention (per frame)');
    }

    // Thumbnails
    if (result.fakeFrames && result.fakeFrames.length > 0) {
      addSection('Top Frame Previews');
      const thumbs = result.fakeFrames.slice(0, 3);
      const thumbSize = 90;
      thumbs.forEach((f, i) => {
        const x = margin + i * (thumbSize + 16);
        if (f.preview) {
          try {
            doc.addImage(f.preview, 'JPEG', x, y, thumbSize, thumbSize);
          } catch {
            doc.setFillColor(...colors.border);
            doc.rect(x, y, thumbSize, thumbSize, 'F');
          }
        } else {
          doc.setFillColor(...colors.border);
          doc.rect(x, y, thumbSize, thumbSize, 'F');
          doc.setFontSize(9);
          doc.setTextColor(...colors.muted);
          doc.text('No preview', x + 10, y + thumbSize / 2);
        }
        doc.setFontSize(10);
        doc.setTextColor(...colors.text);
        doc.text(`Frame ${f.index}`, x, y + thumbSize + 12);
        doc.text(`${(f.score * 100).toFixed(1)}%`, x, y + thumbSize + 24);
      });
      y += thumbSize + 34;
    }

    // Detected anomalies
    addSection('Detected Anomalies');
    if (result.detectedAnomalies.length === 0) {
      doc.setFontSize(11);
      doc.setTextColor(...colors.muted);
      doc.text('None noted', margin, y);
      y += 14;
    } else {
      doc.setFontSize(11);
      doc.setTextColor(...colors.text);
      doc.setFillColor(...colors.accent);
      result.detectedAnomalies.forEach((anomaly) => {
        doc.circle(margin + 3, y - 3, 2, 'F');
        doc.text(anomaly, margin + 12, y);
        y += 14;
      });
    }

    // Footer
    const footerY = Math.max(y + 24, pageHeight - 60);
    doc.setDrawColor(...colors.border);
    doc.line(margin, footerY, pageWidth - margin, footerY);
    doc.setFontSize(9);
    doc.setTextColor(...colors.muted);
    const footerText = `Model: ${result.modelName || 'Unknown'}  |  Threshold: ${result.threshold ?? 50}%  |  Score Source: ${formatScoreSource(result.scoreSource) || '--'}  |  Version 1.0`;
    doc.text(footerText, margin, footerY + 16);

    const filename = `${result.fileName.replace(/\.[^/.]+$/, '') || 'analysis'}-report.pdf`;
    doc.save(filename);
  };

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <Button variant="ghost" onClick={onBack} className="mb-4">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
        <h1 className="text-foreground mb-2">Analysis Results</h1>
        <p className="text-muted-foreground">{result.fileName}</p>
      </div>

      {/* Main Result Card */}
      <Card className="shadow-lg mb-6">
        <CardContent className="pt-6">
          <div className="text-center py-8">
            {/* File Preview */}
            {result.filePreview && (
              <div className="mb-6">
                {result.type === 'video' ? (
                  <video 
                    src={result.filePreview} 
                    controls 
                    className="max-w-md mx-auto rounded-lg shadow-md max-h-64 object-contain"
                  />
                ) : (
                  <img 
                    src={result.filePreview} 
                    alt={result.fileName}
                    className="max-w-md mx-auto rounded-lg shadow-md max-h-64 object-contain"
                  />
                )}
              </div>
            )}
            {result.isDeepfake ? (
              <AlertTriangle className="w-20 h-20 mx-auto mb-4 text-red-600" />
            ) : (
              <CheckCircle className="w-20 h-20 mx-auto mb-4 text-green-600" />
            )}
            <h2 className="text-foreground mb-2">
              {result.isDeepfake ? 'Deepfake Detected' : 'Authentic Media'}
            </h2>
            <Badge
              variant={result.isDeepfake ? 'destructive' : 'default'}
              className={`text-lg px-6 py-2 ${!result.isDeepfake ? 'bg-green-600 hover:bg-green-700' : ''}`}
            >
              {result.isDeepfake ? 'FAKE' : 'REAL'}
            </Badge>
            <div className="mt-6">
              <p className="text-sm text-muted-foreground mb-2">Confidence Score</p>
              <p className="text-foreground">{result.confidence}%</p>
              <Progress value={result.confidence} className="h-3 mt-2 max-w-md mx-auto" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Show detailed analysis only when deepfake is detected */}
      {result.isDeepfake && (
        <>
          {/* Detection Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card className="shadow-md bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-muted-foreground">Confidence</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-foreground">{result.confidence}%</p>
                <Progress value={result.confidence} className="h-2 mt-2" />
              </CardContent>
            </Card>
            <Card className="shadow-md bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-muted-foreground">Fake Probability</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-foreground">{result.probability}%</p>
                <Progress value={result.probability} className="h-2 mt-2" />
              </CardContent>
            </Card>
            <Card className="shadow-md bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-muted-foreground">Threshold</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-foreground">{result.threshold ?? 50}%</p>
                <Progress value={result.threshold ?? 50} className="h-2 mt-2" />
              </CardContent>
            </Card>
          </div>

          {/* Detailed Analysis */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Detected Anomalies */}
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle>Detected Inconsistencies</CardTitle>
                <CardDescription>
                  Analysis artifacts and anomalies found in the media
                </CardDescription>
              </CardHeader>
              <CardContent>
                {result.detectedAnomalies.length > 0 ? (
                  <ul className="space-y-2">
                    {result.detectedAnomalies.map((anomaly, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <div className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0" />
                        <span className="text-sm text-foreground">{anomaly}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No significant anomalies detected</p>
                )}
              </CardContent>
            </Card>

            {/* Technical Details */}
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle>Technical Analysis</CardTitle>
                <CardDescription>
                  GAN fingerprinting and spatiotemporal analysis results
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-muted-foreground">GAN Fingerprint</span>
                  <Badge variant={result.ganFingerprint ? 'destructive' : (result.ganScore && result.ganScore > 30) ? 'secondary' : 'outline'}>
                    {result.ganFingerprint ? 'Detected' : (result.ganScore && result.ganScore > 30) ? 'Weak Signal' : 'Not Found'}
                  </Badge>
                </div>
                {(result.ganScore !== undefined || result.maxFrameScore !== undefined) && (
                  <div className="flex items-center justify-between py-2 border-b border-border">
                    <span className="text-sm text-muted-foreground">GAN Signal</span>
                    <span className="text-foreground">
                      {result.ganScore !== undefined ? `${result.ganScore}%` : '--'}
                      {result.maxFrameScore !== undefined ? ` (peak ${result.maxFrameScore}%)` : ''}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-muted-foreground">GAN Family</span>
                  <span className="text-foreground">
                    {result.ganFamily || 'Not available'}
                    {result.ganFamilyConfidence !== undefined && result.ganFamilyConfidence !== null && (
                      <span className="text-muted-foreground ml-1">
                        ({result.ganFamilyConfidence.toFixed(1)}%)
                      </span>
                    )}
                  </span>
                </div>
                {result.familyProbs && Object.keys(result.familyProbs).length > 0 && (
                  <div className="py-2 border-b border-border">
                    <span className="text-sm text-muted-foreground block mb-2">Family Probabilities</span>
                    <div className="space-y-1">
                      {Object.entries(result.familyProbs)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .map(([family, prob]) => (
                          <div key={family} className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">{family}</span>
                            <span className="text-foreground">{(prob as number).toFixed(1)}%</span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-muted-foreground">Media Type</span>
                  <span className="text-foreground">{result.type.toUpperCase()}</span>
                </div>
                {result.spatiotemporalScore !== undefined && (
                  <div className="flex items-center justify-between py-2 border-b border-border">
                    <span className="text-sm text-muted-foreground">Spatiotemporal Score</span>
                    <span className="text-foreground">{result.spatiotemporalScore}%</span>
                  </div>
                )}
                {result.scoreSource && formatScoreSource(result.scoreSource) && (
                  <div className="flex items-center justify-between py-2">
                    <span className="text-sm text-muted-foreground">Score Source</span>
                    <span className="text-foreground capitalize">{formatScoreSource(result.scoreSource)}</span>
                  </div>
                )}
                {result.frameScores && result.frameScores.length > 0 && (
                  <div className="pt-2">
                    <p className="text-sm text-muted-foreground mb-2">Top suspicious frames</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                      {result.frameScores
                        .map((score, idx) => ({ score, idx }))
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 5)
                        .map(({ score, idx }) => (
                          <div key={idx} className="flex items-center justify-between gap-3 text-sm text-foreground">
                            <div className="flex-1 flex items-center justify-between">
                              <span>Frame {idx + 1}</span>
                              <span>{(score * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
                {result.fakeFrames && result.fakeFrames.length > 0 && (
                  <div className="pt-2">
                    <p className="text-sm text-muted-foreground mb-2">Preview of top frames</p>
                    <div className="grid grid-cols-1 gap-3">
                      {result.fakeFrames.slice(0, 3).map((f, i) => (
                        <div key={`${f.index}-${i}`} className="flex items-center gap-3">
                          {f.preview ? (
                            <img
                              src={f.preview}
                              alt={`Frame ${f.index}`}
                              className="w-20 h-20 object-cover rounded border"
                            />
                          ) : (
                            <div className="w-20 h-20 rounded border border-border flex items-center justify-center text-xs text-muted-foreground">
                              No preview
                            </div>
                          )}
                          <div className="text-sm text-foreground">
                            <div>Frame {f.index}</div>
                            <div>{(f.score * 100).toFixed(1)}% fake prob</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Download Report */}
          <Card className="shadow-md bg-card border-border">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-foreground mb-1">Export Analysis Report</p>
                  <p className="text-sm text-muted-foreground">
                    Download a detailed PDF report with all analysis results and metrics
                  </p>
                </div>
                <Button onClick={downloadReport} className="bg-primary hover:bg-primary/90">
                  <Download className="w-4 h-4 mr-2" />
                  Download Report
                </Button>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
