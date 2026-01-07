import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Shield, Target, Users, Brain, Zap } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';

export function AboutPage() {
  const teamMembers = [
    { name: 'Mrs.Prathibha', role: 'Project Guide', initials: 'PA' },
    { name: 'Chethana R', role: 'Lead Researcher', initials: 'CR' },
    { name: 'Shriya R , A Anne Serena', role: 'ML Engineers', initials: 'SA' },
    { name: 'Bharath M', role: 'Data Scientist', initials: 'BM' },
  ];

  const techniques = [
    {
      icon: Brain,
      title: 'Spatiotemporal Analysis',
      description: 'Analyzes temporal inconsistencies in video sequences by examining frame-to-frame variations and detecting unnatural motion patterns that are characteristic of deepfake generation.'
    },
    {
      icon: Zap,
      title: 'GAN Fingerprinting',
      description: 'Identifies unique artifacts left by Generative Adversarial Networks (GANs) during the synthesis process. Each GAN architecture leaves distinct fingerprints that can be detected.'
    },
  ];

  return (
    <div className="max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="mb-8 text-center">
        <div className="w-16 h-16 bg-primary rounded-2xl flex items-center justify-center mx-auto mb-4">
          <Shield className="w-10 h-10 text-white" />
        </div>
        <h1 className="text-foreground mb-2">
          About Our System
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          An advanced AI-powered solution for detecting deepfake media using cutting-edge machine learning techniques
        </p>
      </div>

      {/* Mission Statement */}
      <Card className="shadow-md mb-8 bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-foreground">
            <Target className="w-5 h-5 text-primary" />
            Mission & Purpose
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-foreground">
            Our deepfake detection system aims to combat the growing threat of synthetic media manipulation by providing researchers, journalists, and organizations with reliable tools to verify media authenticity.
          </p>
          <p className="text-foreground">
            In an era where deepfake technology is becoming increasingly sophisticated, our system leverages state-of-the-art machine learning algorithms to identify manipulated images and videos with high accuracy, helping to maintain trust in digital media.
          </p>
          <p className="text-foreground">
            This academic research project demonstrates the practical application of advanced AI techniques in addressing real-world challenges related to digital media integrity and information security.
          </p>
        </CardContent>
      </Card>

      {/* Technical Approach */}
      <div className="mb-8">
        <h2 className="text-foreground mb-4">Technical Approach</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {techniques.map((technique, index) => {
            const Icon = technique.icon;
            return (
              <Card key={index} className="shadow-md bg-card border-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-foreground">
                    <Icon className="w-5 h-5 text-primary" />
                    {technique.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{technique.description}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Key Features */}
      <Card className="shadow-md mb-8 bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground">Key Features</CardTitle>
          <CardDescription>
            Advanced capabilities of our detection system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="text-foreground mb-1">Multi-Modal Detection</p>
                <p className="text-sm text-muted-foreground">Supports both image and video analysis</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="text-foreground mb-1">High Accuracy</p>
                <p className="text-sm text-muted-foreground">96.8% average detection accuracy</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="text-foreground mb-1">Real-time Processing</p>
                <p className="text-sm text-muted-foreground">Fast analysis with detailed reporting</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="text-foreground mb-1">Comprehensive Reports</p>
                <p className="text-sm text-muted-foreground">Detailed metrics and visualizations</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Team Section */}
      <Card className="shadow-md bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-foreground">
            <Users className="w-5 h-5 text-primary" />
            Research Team
          </CardTitle>
          <CardDescription>
            Dedicated researchers and developers behind this project
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {teamMembers.map((member, index) => (
              <div key={index} className="flex items-center gap-4 p-4 bg-secondary/50 rounded-lg">
                <Avatar className="w-12 h-12">
                  <AvatarFallback className="bg-primary text-white">
                    {member.initials}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <p className="text-foreground">{member.name}</p>
                  <p className="text-sm text-muted-foreground">{member.role}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
