import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Sparkles, Camera, ArrowLeft, CheckCircle, XCircle, Brain, Waves, Volume2, Palette, SquareStack } from 'lucide-react';
import { Progress } from './ui/progress';

interface IndividualCheck {
  method: string;
  is_ai: boolean;
  confidence: number;
  score: number;
  details?: string;
  ai_probability?: number;
  natural_probability?: number;
}

interface AIGeneratedResult {
  fileName: string;
  prediction: 'AI_Generated' | 'Natural';
  probability: number;
  confidence: number;
  final_score: number;
  confidence_level: string;
  votes: {
    ai: number;
    natural: number;
    total: number;
  };
  individual_checks: IndividualCheck[];
  analysis_summary: {
    result: string;
    confidence: string;
    checks: string[];
  };
  filePreview?: string;
}

interface AIGeneratedResultsPageProps {
  result: AIGeneratedResult | null;
  onBack: () => void;
}

const getCheckIcon = (method: string) => {
  switch (method) {
    case 'Neural Network':
      return Brain;
    case 'Frequency Analysis':
      return Waves;
    case 'Noise Analysis':
      return Volume2;
    case 'Color Distribution':
      return Palette;
    case 'Edge Analysis':
      return SquareStack;
    default:
      return CheckCircle;
  }
};

export function AIGeneratedResultsPage({ result, onBack }: AIGeneratedResultsPageProps) {
  if (!result) {
    return (
      <div className="max-w-4xl mx-auto">
        <Card className="shadow-md bg-card border-border">
          <CardContent className="pt-6 text-center py-12">
            <Sparkles className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">No AI-Generated detection results available</p>
            <p className="text-sm text-muted-foreground mt-2">Upload an image to begin analysis</p>
            <Button onClick={onBack} className="mt-6 bg-purple-600 hover:bg-purple-700">
              Back to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const isAIGenerated = result.prediction === 'AI_Generated';

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <Button variant="ghost" onClick={onBack} className="mb-4">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
        <h1 className="text-foreground mb-2">AI-Generated Detection Results</h1>
        <p className="text-muted-foreground">{result.fileName}</p>
      </div>

      {/* Main Result Card */}
      <Card className="shadow-lg mb-6 bg-card border-border">
        <CardContent className="pt-6">
          <div className="text-center py-8">
            {/* File Preview */}
            {result.filePreview && (
              <div className="mb-6">
                <img 
                  src={result.filePreview} 
                  alt={result.fileName}
                  className="max-w-md mx-auto rounded-lg shadow-md max-h-64 object-contain"
                />
              </div>
            )}
            {isAIGenerated ? (
              <div className="w-20 h-20 mx-auto mb-4 bg-purple-600 rounded-full flex items-center justify-center">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
            ) : (
              <div className="w-20 h-20 mx-auto mb-4 bg-green-600 rounded-full flex items-center justify-center">
                <Camera className="w-10 h-10 text-white" />
              </div>
            )}
            <h2 className="text-foreground text-2xl font-bold mb-2">
              {isAIGenerated ? '🤖 AI-Generated Image' : '📸 Natural/Real Photo'}
            </h2>
            <p className="text-muted-foreground mb-4">
              {isAIGenerated 
                ? 'This image appears to be created by AI tools (DALL-E, Midjourney, Stable Diffusion, etc.)'
                : 'This image appears to be a natural photograph taken by a camera'
              }
            </p>
            <Badge
              className={`text-lg px-6 py-2 ${
                isAIGenerated 
                  ? 'bg-purple-600 hover:bg-purple-700' 
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isAIGenerated ? 'AI-GENERATED' : 'NATURAL'}
            </Badge>
            <div className="mt-6">
              <p className="text-sm text-muted-foreground mb-2">Confidence Score</p>
              <p className="text-3xl font-bold text-foreground">{(result.confidence * 100).toFixed(1)}%</p>
              <Progress value={result.confidence * 100} className="h-3 mt-2 max-w-md mx-auto" />
              <p className="text-xs text-muted-foreground mt-2">Confidence Level: {result.confidence_level}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Show detailed analysis only when AI-Generated is detected */}
      {isAIGenerated && (
        <>
          {/* Voting Summary */}
          <Card className="shadow-md mb-6 bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">Analysis Consensus</CardTitle>
              <CardDescription>How the 5 detection methods voted</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className={`p-4 rounded-lg text-center ${result.votes.ai > result.votes.natural ? 'bg-purple-600/20 border-2 border-purple-600' : 'bg-secondary/50'}`}>
                  <Sparkles className="w-8 h-8 mx-auto mb-2 text-purple-500" />
                  <p className="text-3xl font-bold text-foreground">{result.votes.ai}</p>
                  <p className="text-sm text-muted-foreground">votes AI-Generated</p>
                </div>
                <div className={`p-4 rounded-lg text-center ${result.votes.natural > result.votes.ai ? 'bg-green-600/20 border-2 border-green-600' : 'bg-secondary/50'}`}>
                  <Camera className="w-8 h-8 mx-auto mb-2 text-green-500" />
                  <p className="text-3xl font-bold text-foreground">{result.votes.natural}</p>
                  <p className="text-sm text-muted-foreground">votes Natural</p>
                </div>
              </div>
              <p className="text-center text-sm text-muted-foreground mt-4">
                {result.votes.ai}/{result.votes.total} methods detected AI generation patterns
              </p>
            </CardContent>
          </Card>

          {/* Individual Checks */}
          <Card className="shadow-md mb-6 bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">Multi-Check Analysis Details</CardTitle>
              <CardDescription>Results from each detection method</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {result.individual_checks.map((check, index) => {
                  const Icon = getCheckIcon(check.method);
                  return (
                    <div 
                      key={index} 
                      className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg border border-border"
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          check.is_ai ? 'bg-purple-600/20' : 'bg-green-600/20'
                        }`}>
                          <Icon className={`w-5 h-5 ${check.is_ai ? 'text-purple-500' : 'text-green-500'}`} />
                        </div>
                        <div>
                          <p className="font-medium text-foreground">{check.method}</p>
                          <p className="text-xs text-muted-foreground">
                            Score: {(check.score * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="text-right">
                          <p className="text-sm text-muted-foreground">Confidence</p>
                          <p className="font-medium text-foreground">{(check.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <Badge 
                          className={check.is_ai 
                            ? 'bg-purple-600 hover:bg-purple-700' 
                            : 'bg-green-600 hover:bg-green-700'
                          }
                        >
                          {check.is_ai ? 'AI' : 'Natural'}
                        </Badge>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Score Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card className="shadow-md bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-muted-foreground">Final Score</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-foreground">{(result.final_score * 100).toFixed(1)}%</p>
                <Progress value={result.final_score * 100} className="h-2 mt-2" />
                <p className="text-xs text-muted-foreground mt-1">Weighted average of all checks</p>
              </CardContent>
            </Card>
            <Card className="shadow-md bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-muted-foreground">AI Probability</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-purple-500">{(result.probability * 100).toFixed(1)}%</p>
                <Progress value={result.probability * 100} className="h-2 mt-2" />
                <p className="text-xs text-muted-foreground mt-1">Likelihood of AI generation</p>
              </CardContent>
            </Card>
            <Card className="shadow-md bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-muted-foreground">Natural Probability</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-green-500">{((1 - result.probability) * 100).toFixed(1)}%</p>
                <Progress value={(1 - result.probability) * 100} className="h-2 mt-2" />
                <p className="text-xs text-muted-foreground mt-1">Likelihood of real photo</p>
              </CardContent>
            </Card>
          </div>

          {/* What This Means */}
          <Card className="shadow-md bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">What This Means</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-muted-foreground">
                <p>
                  <strong className="text-foreground">🤖 AI-Generated Image Detected:</strong> Our multi-check analysis 
                  indicates this image was likely created using AI image generation tools such as DALL-E, Midjourney, 
                  Stable Diffusion, or similar technologies.
                </p>
                <p>
                  The analysis found patterns typical of AI-generated images including unusual frequency distributions, 
                  uniform noise patterns, and characteristic edge smoothness that differ from natural photographs.
                </p>
                <p className="text-sm">
                  <em>Note: This analysis is based on statistical patterns and may not be 100% accurate. 
                  Always use human judgment alongside automated detection.</em>
                </p>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

