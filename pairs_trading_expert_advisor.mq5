//+------------------------------------------------------------------+
//|                      PairsResidualEA_ZLiveExec.mq5               |
//| Rolling OLS (beta) residual pairs trading with z-bands           |
//| - β/SMA/SD from last closed bar window (no look-ahead)           |
//| - ENTRY & EXIT triggered by z_live intrabar                      |
//| - Bands/HUD drawing unchanged, updated on new bars               |
//+------------------------------------------------------------------+
#property strict
#property version   "1.8"
#include <Trade/Trade.mqh>

//------------------------------ Inputs ------------------------------//
input string           InpSymbol1    = "NVDA.NAS";
input string           InpSymbol2    = "AMD.NAS";
input ENUM_TIMEFRAMES  InpTF         = PERIOD_M30;

input int              BetaWindow    = 200;    // lookback for β (past-only)
input int              BandWindow    = 200;    // window for SMA/Std on s = y - βx
input double           Z_in          = 2.0;    // enter when |z_live| >= Z_in
input double           Z_out         = 0.30;   // exit when |z_live| < Z_out
input double           BaseLots1     = 0.10;   // user lot for Symbol1; Symbol2 β-dollar-neutral
input int              SlippagePts   = 10;
input int              Magic         = 7201502;

input int              DrawLookback  = 500;
input color            ColSpread     = clrDodgerBlue;
input color            ColSMA        = clrSilver;
input color            ColUpper      = clrTomato;
input color            ColLower      = clrMediumSeaGreen;

// HUD placement (avoid being hidden by the Data Window)
input ENUM_BASE_CORNER HudCorner     = CORNER_LEFT_UPPER;
input int              HudX          = 10;
input int              HudY          = 18;
input int              HudFontSize   = 10;
input color            HudColor      = clrWhite;

//---------------------------- Globals -------------------------------//
CTrade trade;
string sym1, sym2;
ENUM_TIMEFRAMES tf;
datetime last_bar_time = 0;

string PREF_SMA    = "PRS_SMA_";
string PREF_UPPER  = "PRS_UPPER_";
string PREF_LOWER  = "PRS_LOWER_";
string PREF_SPREAD = "PRS_SPREAD_";
string OBJ_HUD     = "PRS_HUD";

//============================= Init =================================//
int OnInit()
{
   sym1 = InpSymbol1; sym2 = InpSymbol2; tf = InpTF;

   if(!SymbolSelect(sym1,true) || !SymbolSelect(sym2,true))
   { Print("SymbolSelect failed."); return(INIT_FAILED); }

   if(BetaWindow<20 || BandWindow<20)   { Print("Increase BetaWindow/BandWindow (>=20)."); return(INIT_FAILED); }
   if(Z_in<=0 || Z_out<=0 || Z_out>=Z_in){ Print("Need Z_in>0 and 0<Z_out<Z_in."); return(INIT_FAILED); }
   if(BaseLots1<=0)                      { Print("BaseLots1 must be > 0."); return(INIT_FAILED); }

   CreateOrResetLabel(OBJ_HUD);
   ObjectSetString(0, OBJ_HUD, OBJPROP_TEXT, "warming up...");
   ChartRedraw();

   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(SlippagePts);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   DeleteByPrefix(PREF_SPREAD);
   DeleteByPrefix(PREF_SMA);
   DeleteByPrefix(PREF_UPPER);
   DeleteByPrefix(PREF_LOWER);
   ObjectDelete(0, OBJ_HUD);
}

//============================= OnTick ===============================//
void OnTick()
{
   // 1) Trade using live z each tick (β/SMA/SD from last closed bar)
   TradeWithLiveZ();

   // 2) Update HUD each tick
   UpdateLiveHud();

   // 3) Only redraw bands on a new closed bar (visuals)
   datetime tbuf[];
   if(CopyTime(sym1, tf, 0, 2, tbuf) < 2) return;
   if(last_bar_time == tbuf[1]) return;
   last_bar_time = tbuf[1];

   DrawBandsAndHud();
   UpdateLiveHud(); // keep HUD live after redraw
}

//======================== Data / Calculations =======================//
int CopyBothCloses(const int count, double &c1[], double &c2[], datetime &times[])
{
   ArraySetAsSeries(c1,true); ArraySetAsSeries(c2,true); ArraySetAsSeries(times,true);
   int n1 = CopyClose(sym1, tf, 0, count, c1);
   int n2 = CopyClose(sym2, tf, 0, count, c2);
   int nt = CopyTime (sym1, tf, 0, count, times);
   return MathMin(n1, MathMin(n2, nt));
}

// β on [shift+1 .. shift+BetaWindow] (past-only), then s,z at 'shift'
bool ComputeAtShiftFromArrays(const int shift,
                              const double &c1[], const double &c2[], const datetime &t[],
                              double &beta, double &spread, double &sma, double &sd, double &z)
{
   beta=spread=sma=sd=z=EMPTY_VALUE;

   const int sz = ArraySize(c1);
   if(sz==0 || ArraySize(c2)!=sz || ArraySize(t)!=sz) return false;

   const int needMax = shift + MathMax(BetaWindow, BandWindow);
   if(needMax >= sz) return false;

   // OLS β with intercept
   const int beg = shift+1, end = shift+BetaWindow;
   double sx=0.0, sy=0.0, sxx=0.0, sxy=0.0;
   for(int i=beg; i<=end; ++i)
   { const double x=c2[i], y=c1[i]; sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; }
   const double w=(double)BetaWindow, mx=sx/w, my=sy/w;
   const double varx=MathMax(1e-12, sxx/w - mx*mx), cov=sxy/w - mx*my;
   beta = cov/varx;

   // spread at target bar
   spread = c1[shift] - beta*c2[shift];

   // stats over [shift .. shift+BandWindow-1]
   double sum=0.0, sumsq=0.0;
   for(int i=shift; i<shift+BandWindow; ++i)
   { const double s=c1[i]-beta*c2[i]; sum+=s; sumsq+=s*s; }
   sma = sum/(double)BandWindow;
   const double var = MathMax(0.0, sumsq/(double)BandWindow - sma*sma);
   sd = MathSqrt(var);
   z  = (sd>0.0 ? (spread - sma)/sd : EMPTY_VALUE);
   return true;
}

// Compute β/SMA/SD from last closed bar (shift=1) + z_last and z_live
bool ComputeLiveParams(double &beta_ref, double &s_last, double &sma, double &sd,
                       double &z_last, double &z_live, double &s_live)
{
   const int need = MathMax(BetaWindow, BandWindow) + 2; // need index 0 & window reach
   double c1[], c2[]; datetime t[];
   int got = CopyBothCloses(need, c1, c2, t);
   if(got < need) return false;

   if(!ComputeAtShiftFromArrays(1, c1, c2, t, beta_ref, s_last, sma, sd, z_last)) return false;

   s_live = c1[0] - beta_ref*c2[0];
   z_live = (sd>0.0 ? (s_live - sma)/sd : EMPTY_VALUE);
   return true;
}

//============================= HUD ==================================//
void UpdateLiveHud()
{
   double b, s_last, sma, sd, z_last, z_live, s_live;
   if(!ComputeLiveParams(b, s_last, sma, sd, z_last, z_live, s_live)) return;

   string zlast = (z_last==EMPTY_VALUE? "-" : DoubleToString(z_last,3));
   string zliv  = (z_live==EMPTY_VALUE? "-" : DoubleToString(z_live,3));

   string hud = StringFormat("β = %.5f\ns (spread) = %.5f   z = %s\nz_live = %s",
                             b, s_last, zlast, zliv);
   ObjectSetString(0, OBJ_HUD, OBJPROP_TEXT, hud);
   ChartRedraw();
}

//============================= Trading ==============================//
// Use z_live for signals on every tick (β/SMA/SD from last closed bar)
void TradeWithLiveZ()
{
   double b, s_last, sma, sd, z_last, z_live, s_live;
   if(!ComputeLiveParams(b, s_last, sma, sd, z_last, z_live, s_live)) return;
   if(z_live==EMPTY_VALUE) return;

   int pos = GetSpreadState(); // +1 long spread, -1 short, 0 flat

   if(pos==0)
   {
      if(z_live >= Z_in)       EnterSpread(-1, b); // short spread
      else if(z_live <= -Z_in) EnterSpread(+1, b); // long spread
   }
   else
   {
      if(MathAbs(z_live) < Z_out) ExitSpread();    // intrabar mean reversion exit
   }
}

int GetSpreadState()
{
   int s1=0, s2=0;
   const int total = PositionsTotal();
   for(int i=0; i<total; ++i)
   {
      ulong tk = PositionGetTicket(i);
      if(tk==0) continue;
      if(!PositionSelectByTicket(tk)) continue;

      if((long)PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      string psym = (string)PositionGetString(POSITION_SYMBOL);
      int    type = (int)PositionGetInteger(POSITION_TYPE);

      if(psym == sym1) s1 = (type==POSITION_TYPE_BUY)? +1 : -1;
      if(psym == sym2) s2 = (type==POSITION_TYPE_BUY)? +1 : -1;
   }
   if(s1==+1 && s2==-1) return +1;
   if(s1==-1 && s2==+1) return -1;
   return 0;
}

void EnterSpread(const int dir, const double beta)
{
   // dir: +1 long spread (buy sym1, sell sym2); -1 short spread (sell sym1, buy sym2)
   double lot1 = BaseLots1;
   double step1 = SymbolInfoDouble(sym1, SYMBOL_VOLUME_STEP);
   double vmin1 = SymbolInfoDouble(sym1, SYMBOL_VOLUME_MIN);
   double vmax1 = SymbolInfoDouble(sym1, SYMBOL_VOLUME_MAX);
   lot1 = MathMax(vmin1, MathMin(vmax1, MathFloor(lot1/step1+0.5)*step1));

   double px1 = 0.5*(SymbolInfoDouble(sym1,SYMBOL_BID)+SymbolInfoDouble(sym1,SYMBOL_ASK));
   double px2 = 0.5*(SymbolInfoDouble(sym2,SYMBOL_BID)+SymbolInfoDouble(sym2,SYMBOL_ASK));
   double cs1 = SymbolInfoDouble(sym1, SYMBOL_TRADE_CONTRACT_SIZE);
   double cs2 = SymbolInfoDouble(sym2, SYMBOL_TRADE_CONTRACT_SIZE);
   if(px1<=0 || px2<=0 || cs1<=0 || cs2<=0){ Print("Price/contract size unavailable."); return; }

   // β-dollar-neutral using |β|
   double lot2 = MathAbs(beta) * lot1 * (cs1*px1)/(cs2*px2);
   double step2 = SymbolInfoDouble(sym2, SYMBOL_VOLUME_STEP);
   double vmin2 = SymbolInfoDouble(sym2, SYMBOL_VOLUME_MIN);
   double vmax2 = SymbolInfoDouble(sym2, SYMBOL_VOLUME_MAX);
   lot2 = MathMax(vmin2, MathMin(vmax2, MathFloor(lot2/step2+0.5)*step2));

   if(lot1<=0 || lot2<=0){ Print("Lot calculation produced zero."); return; }

   trade.SetDeviationInPoints(SlippagePts);
   bool ok1=false, ok2=false;
   if(dir>0) { ok1=trade.Buy(lot1, sym1);  ok2=trade.Sell(lot2, sym2); }
   else      { ok1=trade.Sell(lot1, sym1); ok2=trade.Buy(lot2, sym2);  }

   if(!ok1 || !ok2)
   {
      Print("EnterSpread error: sym1=",ok1," sym2=",ok2," ret=",trade.ResultRetcode()," ",trade.ResultRetcodeDescription());
      if(ok1 && !ok2) CloseSymbol(sym1);
      if(!ok1 && ok2) CloseSymbol(sym2);
   }
}

void ExitSpread(){ CloseSymbol(sym1); CloseSymbol(sym2); }

void CloseSymbol(const string symbol)
{
   const int total = PositionsTotal();
   for(int i=0; i<total; ++i)
   {
      ulong tk = PositionGetTicket(i);
      if(tk==0) continue;
      if(!PositionSelectByTicket(tk)) continue;

      if((long)PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      if((string)PositionGetString(POSITION_SYMBOL) != symbol) continue;

      trade.SetDeviationInPoints(SlippagePts);
      trade.PositionClose(tk);
   }
}

//============================= Drawing ==============================//
void DeleteByPrefix(const string prefix)
{
   int total = ObjectsTotal(0, 0, -1);
   for(int i=total-1; i>=0; --i)
   {
      string name = ObjectName(0, i, 0);
      if(StringLen(name)==0) continue;
      if(StringFind(name, prefix, 0) == 0)
         ObjectDelete(0, name);
   }
}

void CreateOrResetLabel(const string name)
{
   if(ObjectFind(0, name)==-1)
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);

   ObjectSetInteger(0, name, OBJPROP_CORNER,    HudCorner);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, HudX);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, HudY);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  HudFontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR,     HudColor);
   ObjectSetString (0, name, OBJPROP_FONT,      "Arial");
}

// Draw one time series as trend segments
void DrawSeriesAsTrends(const string prefix, const color col, const int width,
                        const datetime &times[], const double &vals[], const int n)
{
   DeleteByPrefix(prefix);
   if(n<=1) return;

   for(int i=0; i<n-1; ++i)
   {
      if(vals[i]==EMPTY_VALUE || vals[i+1]==EMPTY_VALUE) continue;
      string name = prefix + "#" + IntegerToString(i);
      ObjectCreate(0, name, OBJ_TREND, 0, times[i], vals[i], times[i+1], vals[i+1]);
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
      ObjectSetInteger(0, name, OBJPROP_RAY, false);
   }
}

void DrawBandsAndHud() 
{
   int need = MathMax(MathMax(BetaWindow, BandWindow)+5, DrawLookback);
   double c1[], c2[]; datetime t[];
   int got = CopyBothCloses(need, c1, c2, t);
   if(got < need) return;

   datetime times[]; ArrayResize(times, DrawLookback);
   double smaA[];    ArrayResize(smaA, DrawLookback);
   double upA[];     ArrayResize(upA,  DrawLookback);
   double loA[];     ArrayResize(loA,  DrawLookback);
   double sA[];      ArrayResize(sA,   DrawLookback);

   int filled=0;
   for(int sh=DrawLookback-1; sh>=0; --sh)
   {
      double b, s, sma, sd, z;
      if(!ComputeAtShiftFromArrays(sh, c1, c2, t, b, s, sma, sd, z)) continue;
      times[filled]=t[sh];
      smaA[filled]=sma;
      sA[filled]=s;
      if(sd>0.0){ upA[filled]=sma + Z_in*sd; loA[filled]=sma - Z_in*sd; }
      else      { upA[filled]=EMPTY_VALUE;   loA[filled]=EMPTY_VALUE;   }
      filled++;
   }
   if(filled<=1) return;

   ArrayResize(times, filled);
   ArrayResize(smaA,  filled);
   ArrayResize(upA,   filled);
   ArrayResize(loA,   filled);
   ArrayResize(sA,    filled);
   ArrayReverse(times); ArrayReverse(smaA); ArrayReverse(upA); ArrayReverse(loA); ArrayReverse(sA);

   DrawSeriesAsTrends(PREF_SMA,    ColSMA,    1, times, smaA, filled);
   DrawSeriesAsTrends(PREF_UPPER,  ColUpper,  1, times, upA,  filled);
   DrawSeriesAsTrends(PREF_LOWER,  ColLower,  1, times, loA,  filled);
   DrawSeriesAsTrends(PREF_SPREAD, ColSpread, 1, times, sA,   filled);

   // Snapshot for last closed bar (live HUD will overwrite on next tick)
   double b0,s0,sma0,sd0,z0;
   string hud="...";
   if(ComputeAtShiftFromArrays(1, c1, c2, t, b0, s0, sma0, sd0, z0))
      hud = StringFormat("β = %.5f\ns (spread) = %.5f   z = %s",
                         b0, s0, (z0==EMPTY_VALUE?"-":DoubleToString(z0,3)));
   ObjectSetString(0, OBJ_HUD, OBJPROP_TEXT, hud);
} 
