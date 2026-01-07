import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { UploadZone } from './UploadZone';
import { StatsCard } from './StatsCard';
import { Shield, Activity, CheckCircle, AlertCircle, TrendingUp, FileCheck, Sparkles, Camera } from 'lucide-react';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';

interface HomePageProps {
  onAnalyze: (file: File, type: 'image' | 'video') => void;
  onAnalyzeAIGenerated?: (file: File) => void;
  analyses?: {
    id: number;
    fileName: string;
    type: 'Image' | 'Video' | 'AI-Check';
    result: 'Real' | 'Fake' | 'AI-Generated' | 'Natural';
    confidence: number;
    date: string;
    time: string;
  }[];
}

export function HomePage({
  onAnalyze,
  onAnalyzeAIGenerated,
  analyses = [],
}: HomePageProps) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [aiGeneratedFile, setAiGeneratedFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzingType, setAnalyzingType] = useState<'image' | 'video' | 'ai-generated' | null>(null);
  const [progress, setProgress] = useState(0);

  const handleAnalyze = async (type: 'image' | 'video' | 'ai-generated') => {
    const file = type === 'image' ? imageFile : type === 'video' ? videoFile : aiGeneratedFile;
    if (!file) return;

    setAnalyzing(true);
    setAnalyzingType(type);
    setProgress(0);

    // Simulate analysis progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            setAnalyzing(false);
            setAnalyzingType(null);
            if (type === 'ai-generated' && onAnalyzeAIGenerated) {
              onAnalyzeAIGenerated(file);
            } else if (type === 'image' || type === 'video') {
              onAnalyze(file, type);
            }
          }, 500);
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  const totalAnalyses = analyses.length;
  const deepfakeCount = analyses.filter((a) => a.result === 'Fake').length;
  const realCount = totalAnalyses - deepfakeCount;
  const avgConfidence =
    totalAnalyses === 0
      ? 0
      : parseFloat(
          (
            analyses.reduce((acc, a) => acc + a.confidence, 0) / totalAnalyses
          ).toFixed(1)
        );
  const recentAnalyses = analyses.slice(0, 5);

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page Title */}
      <div className="text-center mb-8">
        <h1 className="text-foreground mb-2">
          AI-Driven Deepfake Detection System
        </h1>
        <p className="text-muted-foreground">
          Using Spatiotemporal Analysis and GAN Fingerprinting
        </p>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatsCard
          title="Total Analyses"
          value={totalAnalyses}
          icon={Activity}
          description="Completed in this session"
        />
        <StatsCard
          title="Average Confidence"
          value={`${avgConfidence}%`}
          icon={TrendingUp}
          description={totalAnalyses ? 'Mean model confidence' : 'Run an analysis to see confidence'}
        />
        <StatsCard
          title="Deepfakes Flagged"
          value={deepfakeCount}
          icon={AlertCircle}
          description={`${totalAnalyses ? ((deepfakeCount / totalAnalyses) * 100).toFixed(1) : 0}% of total`}
        />
        <StatsCard
          title="Authentic Media"
          value={realCount}
          icon={CheckCircle}
          description={`${totalAnalyses ? ((realCount / totalAnalyses) * 100).toFixed(1) : 0}% of total`}
        />
      </div>

      {/* Upload Sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Image Upload */}
        <Card className="shadow-md border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Shield className="w-5 h-5 text-primary" />
              Image Analysis
            </CardTitle>
            <CardDescription>
              Upload an image to detect GAN fingerprints and manipulation artifacts
            </CardDescription>
          </CardHeader>
          <CardContent>
            <UploadZone type="image" onFileSelect={setImageFile} />
            <Button
              onClick={() => handleAnalyze('image')}
              disabled={!imageFile || analyzing}
              className="w-full mt-4 bg-primary hover:bg-primary/90"
            >
              Analyze Image
            </Button>
          </CardContent>
        </Card>

        {/* Video Upload */}
        <Card className="shadow-md border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Shield className="w-5 h-5 text-primary" />
              Video Analysis
            </CardTitle>
            <CardDescription>
              Upload a video for spatiotemporal deepfake detection analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <UploadZone type="video" onFileSelect={setVideoFile} />
            <Button
              onClick={() => handleAnalyze('video')}
              disabled={!videoFile || analyzing}
              className="w-full mt-4 bg-primary hover:bg-primary/90"
            >
              Analyze Video
            </Button>
          </CardContent>
        </Card>
      </div>

     

      {/* Analysis Progress */}
      {analyzing && (
        <Card className="mb-8 shadow-md border-border">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4 mb-3">
              <Activity className={`w-5 h-5 animate-pulse ${analyzingType === 'ai-generated' ? 'text-purple-500' : 'text-primary'}`} />
              <p className="text-foreground">
                {analyzingType === 'ai-generated' ? 'AI-Generated detection' : 'Deepfake analysis'} in progress...
              </p>
            </div>
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground mt-2">{progress}% complete</p>
          </CardContent>
        </Card>
      )}

      {/* Recent Analyses */}
      <Card className="shadow-md border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-foreground">
            <FileCheck className="w-5 h-5 text-primary" />
            Recent Analysis Summary
          </CardTitle>
          <CardDescription>Your most recent deepfake detection results</CardDescription>
        </CardHeader>
        <CardContent>
          {recentAnalyses.length === 0 ? (
            <p className="text-sm text-muted-foreground">No analyses yet. Upload an image or video to see live stats.</p>
          ) : (
            <div className="space-y-3">
              {recentAnalyses.map((analysis) => (
                <div
                  key={analysis.id}
                  className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg hover:bg-secondary transition-colors"
                >
                  <div className="flex-1">
                    <p className="text-foreground">{analysis.fileName}</p>
                    <p className="text-sm text-muted-foreground">
                      {analysis.type} - {analysis.date} {analysis.time ? `- ${analysis.time}` : ''}
                    </p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">Confidence</p>
                      <p className="text-foreground">{analysis.confidence}%</p>
                    </div>
                    <Badge
                      variant={
                        analysis.result === 'Fake' || analysis.result === 'AI-Generated' 
                          ? 'destructive' 
                          : 'default'
                      }
                      className={
                        analysis.result === 'Fake' 
                          ? '' 
                          : analysis.result === 'AI-Generated'
                            ? 'bg-purple-600 hover:bg-purple-700'
                            : 'bg-green-600 hover:bg-green-700'
                      }
                    >
                      {analysis.result}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
